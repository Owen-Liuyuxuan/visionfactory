import streamlit as st
from PIL import Image
import onnxruntime as ort
from segmentation.evaluation.labels import PALETTE
import numpy as np
from numba import jit
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def normalize_image(image, rgb_mean = np.array([0.485, 0.456, 0.406]), rgb_std  = np.array([0.229, 0.224, 0.225])):
    image = image.astype(np.float32)
    image = image / 255.0
    image = image - rgb_mean
    image = image / rgb_std
    return image

MONO3D_NAMES = ['car', 'truck', 'bus', 
                'trailer', 'construction_vehicle',
                'pedestrian', 'motorcycle', 'bicycle',
                'traffic_cone', 'barrier']

COLOR_MAPPINGS = {
    'car' : (  0,  0,142),  'truck': (  0,  0, 70) ,
    'bus': (  0, 60,100), 'trailer': (  0,  0,110),
    'construction_vehicle':  (  0,  0, 70), 'pedestrian': (220, 20, 60),
    'motorcycle': (  0,  0,230), 'bicycle': (119, 11, 32),
    'traffic_cone': (180,165,180), 'barrier': (190,153,153)
}

def depth_image_to_point_cloud_array(depth_image, K, rgb_image=None, mask=None):
    """  convert depth image into color pointclouds [xyzbgr]
    
    """
    depth_image = np.copy(depth_image)
    w_range = np.arange(0, depth_image.shape[1], dtype=np.float32)
    h_range = np.arange(0, depth_image.shape[0], dtype=np.float32)
    w_grid, h_grid = np.meshgrid(w_range, h_range) #[H, W]
    K_expand = np.eye(4)
    K_expand[0:3, 0:3] = K
    K_inv = np.linalg.inv(K_expand) #[4, 4]

    #[H, W, 4, 1]
    expand_image = np.stack([w_grid * depth_image, h_grid * depth_image, depth_image, np.ones_like(depth_image)], axis=2)[...,np.newaxis]

    pc_3d = np.matmul(K_inv, expand_image)[..., 0:3, 0] #[H, W, 3]
    if rgb_image is not None:
        pc_3d = np.concatenate([pc_3d, rgb_image], axis=2).astype(np.float32)
    if mask is not None:
        point_cloud = pc_3d[(mask > 0) * (depth_image > 0),:]
    else:
        point_cloud = pc_3d[depth_image > 0,:]
    
    return point_cloud

@jit(nopython=True, cache=True)
def ColorizeSeg(pred_seg, rgb_image, opacity=1.0, palette=PALETTE):
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    h, w = pred_seg.shape
    for i in range(h):
        for j in range(w):
            color_seg[i, j] = palette[pred_seg[i, j]]
    new_image = rgb_image * (1 - opacity) + color_seg * opacity
    new_image = new_image.astype(np.uint8)
    return new_image


@st.cache_resource()
def load_model():
    onnx_path_dict = {
        "mono3d": "/home/yliuhb/vision_collection/model/det3d/dla34_deform_576_768.onnx",
        "seg": "/home/yliuhb/vision_collection/model/segmentation/bisenetv1.onnx",
        "depth": "/home/yliuhb/vision_collection/model/monodepth/monodepth_res101_384_1280.onnx"
    }
    model_dict = {}
    for key in onnx_path_dict.keys():
        onnx_path = onnx_path_dict[key]
        providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '0', 'device_id': str(0)})]
        sess_options = ort.SessionOptions()
        ort_session = ort.InferenceSession(onnx_path, providers=providers, sess_options=sess_options)
        input_shape = ort_session.get_inputs()[0].shape # [1, 3, h, w]
        inference_h = input_shape[2]
        inference_w = input_shape[3]
        model_dict[key] = {
            "ort_session": ort_session,
            "inference_h": inference_h,
            "inference_w": inference_w
        }
    return model_dict

from skimage.morphology import erosion, square

def create_mask_from_segmentation(segmentation, erosion_size=9):
    """
    Create a mask from a segmentation output where each class of pixels in the image is eroded.

    Args:
    - segmentation (np.array): The segmentation image with shape [H, W], containing class indexes.
    - erosion_size (int): The size of the structuring element used for erosion.

    Returns:
    - mask (np.array): The combined mask for all classes.
    """

    unique_classes = np.unique(segmentation)
    combined_mask = np.zeros_like(segmentation)

    for class_idx in unique_classes:
        # Create a mask for the current class
        class_mask = (segmentation == class_idx).astype(np.uint8)

        # Erode the mask
        if erosion_size > 0:
            class_mask = erosion(class_mask, square(erosion_size))

        # Add the eroded mask to the combined mask
        combined_mask = np.maximum(combined_mask, class_mask)

    return combined_mask

def resize(image, model_dict, P=None):
    inter_dict = {}
    inference_h = model_dict['inference_h']
    inference_w = model_dict['inference_w']
    h0, w0 = image.shape[0:2]
    inter_dict['h0'] = h0
    inter_dict['w0'] = w0
    scale = min(inference_h / h0, inference_w / w0)
    inter_dict['scale'] = scale
    h_eff = int(h0 * scale)
    w_eff = int(w0 * scale)
    inter_dict['h_eff'] = h_eff
    inter_dict['w_eff'] = w_eff

    final_image = np.zeros([inference_h, inference_w, 3])
    final_image[0:h_eff, 0:w_eff] = cv2.resize(image, (w_eff, h_eff),
                                                interpolation=cv2.INTER_LINEAR)
    if P is not None:
        P = P.copy()
        P[0:2, :] = P[0:2, :] * scale
        inter_dict['P'] = P
    else:
        inter_dict['P'] = None
    return final_image, inter_dict

def deresize(seg, inter_dict):
    seg = seg[0:inter_dict['h_eff'], 0:inter_dict['w_eff']]
    seg = cv2.resize(seg, (inter_dict['w0'], inter_dict['h0']),
                     interpolation=cv2.INTER_NEAREST)
    return seg

def calibration(fx, fy, cx, cy, k1, k2, p1, p2, image):
    if fx == 0:
        h, w = image.shape[0:2]
        ## Create a default P matrix with hidth and width assuming fx = fy = sqrt(h^2 + w^2)
        _fx = _fy = np.sqrt(h ** 2 + w ** 2)
        _cx = w / 2
        _cy = h / 2
        P = np.array([[_fx, 0, _cx, 0], [0, _fy, _cy, 0], [0, 0, 1, 0]])
        return P, image.copy()
    else:
        P = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]])
        if k1 != 0:
            dist = np.array([k1, k2, p1, p2, 0])
            h, w = image.shape[0:2]
            undistorted_image = cv2.undistort(image, P[0:3, 0:3], dist, None, P[0:3, 0:3])
        else:
            undistorted_image = image.copy()
        return P, undistorted_image

def draw_bbox2d_to_image(image, bboxes2d, color=(255, 0, 255)):
    drawed_image = image.copy()
    for box2d in bboxes2d:
        cv2.rectangle(drawed_image, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), color, 3)
    return drawed_image

def draw_3d_point_clout(point_cloud):
    xyz = point_cloud[:, 0:3]
    rgb = point_cloud[:, 3:6]
    fig = go.Figure(data=[go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=['rgb({},{},{})'.format(r, g, b) for r, g, b in rgb],  # Assigning colors to points
            opacity=1.0,
            symbol='square'
        )
    )])
    fig.update_layout(
        scene=dict(
            aspectmode='data'  # Forces equal aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig

@st.cache_data
def inferencing(_models, image, P):
    for method in ['mono3d', 'seg', 'depth']:
        if method == 'mono3d':
            resized_image, inter_dict = resize(image, _models[method], P)
            P_numpy = np.array(inter_dict['P'], dtype=np.float32)[None]
            input_numpy = np.ascontiguousarray(np.transpose(normalize_image(resized_image), (2, 0, 1))[None], dtype=np.float32)
            outputs = _models[method]['ort_session'].run(None, {'image': input_numpy, 'P2': P_numpy})
            scores = np.array(outputs[0]) # N
            bboxes = np.array(outputs[1]) # N, 12
            cls_indexes = outputs[2] # N
            cls_names = [MONO3D_NAMES[cls_index] for cls_index in cls_indexes]
            bbox2d = bboxes[:, 0:4] / inter_dict['scale'] # N, 4

            rgb_image = image.copy()
            for i in range(scores.shape[0]):
                top_left = (int(bbox2d[i, 0].item()), int(bbox2d[i, 1].item()))
                cls_name = cls_names[i]
                color = COLOR_MAPPINGS[cls_name]
                rgb_image = draw_bbox2d_to_image(rgb_image, bbox2d[i:i+1], color=color)
                cv2.putText(rgb_image, f'{cls_name[:3]}', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            results['mono3d'] = {
                'scores': scores,
                'bboxes': bboxes,
                'cls_indexes': cls_indexes,
                'cls_names': cls_names,
                'drawed_detection_image': rgb_image
            }
        if method == 'seg':
            resized_image, inter_dict = resize(image, _models[method])
            input_numpy = np.ascontiguousarray(np.transpose(normalize_image(resized_image), (2, 0, 1))[None], dtype=np.float32)
            outputs = _models[method]['ort_session'].run(None, {'input': input_numpy})
            seg = outputs[0][0]
            output = deresize(np.array(seg, np.uint8), inter_dict)
            results['seg'] = {
                'seg': output,
            }
        if method == 'depth':
            resized_image, inter_dict = resize(image, _models[method], P)
            input_numpy = np.ascontiguousarray(np.transpose(normalize_image(resized_image), (2, 0, 1))[None], dtype=np.float32)
            P_numpy = np.array(inter_dict['P'], dtype=np.float32)[None]
            outputs = _models[method]['ort_session'].run(None, {'image': input_numpy, 'P2': P_numpy})
            output_depth = deresize(outputs[0][0, 0], inter_dict)
            results['depth'] = {
                'depth': output_depth
            }
    return results




### Starting Webpage

st.title('Vision Demo')

with st.form(key='calibration'):
    col1, col2 = st.columns(2)  # Creates two columns

    with col1:  # First column
        fx = st.number_input('fx', value=0)
        fy = st.number_input('fy', value=0)
        cx = st.number_input('cx', value=0)
        cy = st.number_input('cy', value=0)

    with col2:  # Second column
        k1 = st.number_input('k1', value=0)
        k2 = st.number_input('k2', value=0)
        p1 = st.number_input('p1', value=0)
        p2 = st.number_input('p2', value=0)

    submit_button = st.form_submit_button(label='Calibration Send')

opacity = st.slider("Select Segmentation Color Opacity", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
erosion_size = st.slider("Select Erosion Size", min_value=0, max_value=15, value=3, step=1)
downsample_ratio = st.slider("Select Downsample Ratio", min_value=1, max_value=64, value=4, step=1)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

models = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    results = {}
    P, image = calibration(fx, fy, cx, cy, k1, k2, p1, p2, image)
    results = inferencing(models, image, P)
    
    ## Draw Detection
    plt.subplot(2, 2, 2)
    plt.imshow(results['mono3d']['drawed_detection_image'])
    plt.axis('off')
    
    ## Draw Segmentation 
    seg_image = ColorizeSeg(results['seg']['seg'], image.copy(), opacity=opacity)
    plt.subplot(2, 2, 3)
    plt.imshow(seg_image)
    plt.axis('off')
    
    ## Draw Depth
    plt.subplot(2, 2, 4)
    output_depth = results['depth']['depth']
    plt.imshow(1 / output_depth, cmap='magma', vmin=1/70, vmax=1/max(output_depth.min(), 2.0))
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    ### Publish PointClouds
    kernel = np.ones((7, 7), np.uint8)
    mask = np.logical_not(results['seg']['seg'] == 23)
    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(np.bool_)
    
    masks = create_mask_from_segmentation(results['seg']['seg'], erosion_size=erosion_size)
    mask = np.logical_and(mask, masks)

    point_cloud = depth_image_to_point_cloud_array(results['depth']['depth'],
                                                   P[0:3, 0:3],
                                                   rgb_image=seg_image, mask=mask)
    point_cloud = point_cloud[::downsample_ratio, :]

    st.plotly_chart(draw_3d_point_clout(point_cloud), use_container_width=True)
