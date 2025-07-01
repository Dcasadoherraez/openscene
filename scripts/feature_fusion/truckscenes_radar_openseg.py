import os
import torch
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper

from truckscenes import TruckScenes
from truckscenes.utils.splits import train_detect, val, test, mini_train, mini_val

def get_args():
    '''Command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on nuScenes.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='test', help='split: "train"| "val" ')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(scene_name, args):
    '''Process one scene.'''

    # short hand
    split = args.split
    img_size = args.img_dim
    data_dir = args.data_dir
    point2img_mapper = args.point2img_mapper
    openseg_model = args.openseg_model
    text_emb = args.text_emb
    output_dir = args.output_dir
    
    os.makedirs(join(args.output_dir, scene_name), exist_ok=True)

    
    scene_path = os.path.join(data_dir, 'trainval', scene_name)
    images_dir = os.path.join(scene_path, 'images')
    labelled_points_path = os.path.join(scene_path, 'labelled_points')
    intrinsics_path = os.path.join(scene_path, 'intrinsics')
    poses_path = os.path.join(scene_path, 'poses')
    
    cam_locs = os.listdir(images_dir)
    
    sorted_labelled_point_paths = sorted(os.listdir(labelled_points_path), key= lambda x : int(x.split(".")[0]))
    all_scans = [join(labelled_points_path, i) for i in sorted_labelled_point_paths]
    
    device = torch.device('cpu')
    
    for idx in tqdm(range(len(all_scans))):
        curr_scan_path = all_scans[idx]
        
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        # Only process points with GT label annotation
        locs_in, _, labels_in, instances_in = torch.load(curr_scan_path)
        # mask_entire = labels_in!=None
        # locs_in = locs_in[mask_entire]
        
        mask_entire = np.ones_like(labels_in, dtype=bool)
        locs_in = locs_in[mask_entire]
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################
        ###################################################

        n_points = locs_in.shape[0]

        if exists(join(output_dir, scene_name, f'{idx}.pt')):
            print(scene_name, f'{idx}.pt' + ' already done!')
            continue

        
        n_points_cur = n_points
        counter = torch.zeros((n_points_cur, 1), device=device)
        sum_features = torch.zeros((n_points_cur, args.feat_dim), device=device)

        vis_id = torch.zeros((n_points_cur, len(cam_locs)), dtype=int, device=device)
        for img_id, cam in enumerate(tqdm(cam_locs)):
            pose_path = join(poses_path, cam, f'{idx}.txt')
            intr_path = join(intrinsics_path, cam, f'{idx}.txt')
            img_dir = join(images_dir, cam, f'{idx}.png')
            
            pose = np.loadtxt(pose_path)
            intr = np.loadtxt(intr_path)

            # calculate the 3d-2d mapping
            mapping = np.ones([n_points_cur, 4], dtype=int)
            mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in[:, :3], depth=None, intrinsic=intr)
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
                continue

            mapping = torch.from_numpy(mapping).to(device)
            mask = mapping[:, 3]
            vis_id[:, img_id] = mask
            feat_2d = extract_openseg_img_feature(
                img_dir, openseg_model, text_emb, img_size=[img_size[1], img_size[0]]).to(device)

            feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

            counter[mask!=0]+= 1
            sum_features[mask!=0] += feat_2d_3d[mask!=0]

        counter[counter==0] = 1e-5
        feat_bank = sum_features/counter
        point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

        mask = torch.zeros(n_points, dtype=torch.bool)
        mask[point_ids] = True
        mask_entire[mask_entire==True] = mask
        
        output_path = join(output_dir, scene_name, f'{idx}.pt')
        torch.save({"feat": feat_bank[mask].half().cpu(),
                    "mask_full": mask_entire}, output_path)

        print(join(output_dir, scene_name, f'{idx}.pt') + ' is saved!')

import os
import numpy as np
import torch
import tensorflow as tf
import multiprocessing as mp
import argparse 

# --- Globals for worker processes (will be distinct in each worker) ---
g_worker_openseg_model = None
g_worker_text_emb = None
g_worker_point2img_mapper = None
# --------------------------------------------------------------------

def init_worker(static_config_dict_for_init):
    """
    Initializer function for each worker process.
    Loads the model and other resources once.
    """
    global g_worker_openseg_model, g_worker_text_emb, g_worker_point2img_mapper
    
    worker_init_args = argparse.Namespace(**static_config_dict_for_init)
    process_pid = os.getpid()
    print(f"[PID {process_pid}] Initializing worker...")

    # 1. Configure GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[PID {process_pid}] GPU memory growth set.")
        except RuntimeError as e:
            print(f"[PID {process_pid}] Error setting GPU memory growth: {e}")

    # 2. Load OpenSeg model
    if worker_init_args.openseg_model_path:
        try:
            g_worker_openseg_model = tf.saved_model.load(
                worker_init_args.openseg_model_path,
                tags=[tf.saved_model.SERVING]
            )
            g_worker_text_emb = tf.zeros([1, 1, worker_init_args.feat_dim])
            print(f"[PID {process_pid}] Model loaded successfully.")
        except Exception as e:
            print(f"[PID {process_pid}] Error loading model: {e}")
            g_worker_openseg_model = None # Ensure it's None if loading failed
            g_worker_text_emb = None
    else:
        g_worker_openseg_model = None
        g_worker_text_emb = None
        print(f"[PID {process_pid}] No model path provided. Model not loaded.")

    # 3. Initialize PointCloudToImageMapper
    try:
        g_worker_point2img_mapper = PointCloudToImageMapper(
            image_dim=worker_init_args.img_dim,
            cut_bound=worker_init_args.cut_num_pixel_boundary
        )
        print(f"[PID {process_pid}] PointCloudToImageMapper initialized.")
    except Exception as e:
        print(f"[PID {process_pid}] Error initializing PointCloudToImageMapper: {e}")
        g_worker_point2img_mapper = None
        
    print(f"[PID {process_pid}] Worker initialization complete.")


def process_scene_task_in_worker(scene_id, static_config_dict_for_task):
    """
    Processes a single scene using the pre-loaded global resources in the worker.
    """
    global g_worker_openseg_model, g_worker_text_emb, g_worker_point2img_mapper
    process_pid = os.getpid()

    if g_worker_openseg_model is None and static_config_dict_for_task.get('openseg_model_path'):
        # This case should ideally not happen if init_worker succeeded and a model path was given.
        # It indicates a problem during worker initialization for model loading.
        error_msg = f"[PID {process_pid}] Error: Model not available for scene {scene_id}. Initialization might have failed."
        print(error_msg)
        return error_msg # Or raise an exception

    # Create an args object for process_one_scene, using worker's global resources
    task_args = argparse.Namespace(**static_config_dict_for_task)
    task_args.openseg_model = g_worker_openseg_model
    task_args.text_emb = g_worker_text_emb
    task_args.point2img_mapper = g_worker_point2img_mapper

    # Call your existing scene processing function
    process_one_scene(scene_id, task_args)
    
    return f"Scene {scene_id} processed by PID {process_pid} (using pre-loaded model)."


def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #### Dataset specific parameters #####
    img_dim = (990, 471)

    args.img_dim = img_dim
    args.cut_num_pixel_boundary = 5 # do not use the features on the image boundary
    args.data_dir = "/shared/data/truckScenes/truckscenes_converted"
    args.output_dir = "/shared/data/truckScenes/truckscenes_converted/openseg_all"
    args.split = "mini_train" # train_detect, val, test, mini_train, mini_val
    args.openseg_model = "/home/daniel/spatial_understanding/benchmarks/openscene/openseg_exported_clip/"
    args.process_id_range = None # scene range to process
    args.feat_dim = 768 # CLIP feature dimension
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare a picklable configuration dictionary to be passed to workers
    args.openseg_model_path = args.openseg_model # Keep the path for workers
    
    # These specific attributes will be managed globally within each worker
    args.openseg_model = None # Ensure no loaded model is passed from main
    args.text_emb = None
    args.point2img_mapper = None

    static_config_dict = vars(args).copy()

    scenes_to_process = train_detect + val

    NUM_WORKERS = 6

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=NUM_WORKERS,
                  initializer=init_worker,
                  initargs=(static_config_dict,)) as pool: # Pass config to initializer
        
        tasks_for_starmap = [(scene_id, static_config_dict) for scene_id in scenes_to_process]
        results = pool.starmap(process_scene_task_in_worker, tasks_for_starmap)

    for result in results:
        print(result)
    print("All scene processing tasks submitted and completed.")

if __name__ == "__main__":
    args = get_args()
    main(args)
    
    

# def main(args):
#     seed = 1457
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)

#     #### Dataset specific parameters #####
#     img_dim = (990, 471)

#     args.img_dim = img_dim
#     args.cut_num_pixel_boundary = 5 # do not use the features on the image boundary
#     args.data_dir = "/shared/data/truckScenes/truckscenes_converted"
#     args.output_dir = "/shared/data/truckScenes/truckscenes_converted/openseg"
#     args.split = "mini_train" # train_detect, val, test, mini_train, mini_val
#     args.openseg_model = "/home/daniel/spatial_understanding/benchmarks/openscene/openseg_exported_clip/"
#     args.process_id_range = None # scene range to process
#     args.feat_dim = 768 # CLIP feature dimension
    
#     os.makedirs(args.output_dir, exist_ok=True)

#     # load the openseg model
#     saved_model_path = args.openseg_model
#     args.text_emb = None
#     if args.openseg_model != '':
#         args.openseg_model = tf2.saved_model.load(saved_model_path,
#                     tags=[tf.saved_model.tag_constants.SERVING],)
#         args.text_emb = tf.zeros([1, 1, args.feat_dim])
#     else:
#         args.openseg_model = None

#     # calculate image pixel-3D points correspondances
#     args.point2img_mapper = PointCloudToImageMapper(
#             image_dim=img_dim,
#             cut_bound=args.cut_num_pixel_boundary)

#     scenes = eval(args.split)
#     scenes = train_detect + val
#     for scene in scenes:
#         process_one_scene(scene, args)


# if __name__ == "__main__":
#     args = get_args()
#     print("Arguments:")
#     print(args)

#     main(args)