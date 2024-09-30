import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换
def point_guided_deformation(image, source_pts, target_pts, alpha, eps):
    image = np.array(image)
    # 获取输入图像的高度和宽度
    h, w, _= image.shape
    #print("h:", h)
    #print("w:", w)
    # 创建一个与输入图像大小相同的空图像，用于存放变形后的结果
    warped_image = np.zeros_like(image)

    # 遍历每个像素位置 (x, y)
    for y in range(h):
        for x in range(w):
            #print("x:", x)
            # 当前像素的位置
            now_pt = np.array([x, y])

            # 矩阵Ai
            a_matrix = np.zeros((len(source_pts), 2, 2))

            # 计算所有源点与当前点的距离
            distances = np.linalg.norm(source_pts - now_pt, axis=1)

            # 计算权重
            weights = np.where(distances > 0, 1 / (distances ** (2 * alpha)), float('inf'))

            # 避免无效计算
            if np.sum(weights) < eps:
                warped_image[y, x] = image[y,x]
                continue

            # 加权质心
            weighted_src_center = np.sum(weights[:, np.newaxis] * source_pts, axis=0) / np.sum(weights)
            weighted_dst_center = np.sum(weights[:, np.newaxis] * target_pts, axis=0) / np.sum(weights)

            # 中心化点
            source_pts_center = source_pts - weighted_src_center
            target_pts_center = target_pts - weighted_dst_center

            # 矩阵C
            c_first_row = now_pt - weighted_src_center
            c_second_row = np.array([c_first_row[1], -c_first_row[0]])
            c = np.array([c_first_row, c_second_row])
            c = c.T

            # 矩阵Ai
            for i, weight in enumerate(weights):
                b_second_row = np.array([source_pts_center[i][1], -source_pts_center[i][0]])
                b = np.array([source_pts_center[i], b_second_row])

                a_matrix[i] = weight * (b @ c)

            # fr(v)
            products = target_pts_center[:, np.newaxis, :] @ a_matrix
            fr_1 = np.sum(products, axis=0)
            fr = ((np.linalg.norm(now_pt - weighted_src_center) * fr_1 )/ np.linalg.norm(fr_1)) + weighted_dst_center
            if np.isnan(fr).any():
                warped_image[y, x] = image[y, x]
                print("fr contains NaN values:", fr)
            else:
                fr_x, fr_y = int(np.round(fr[0][0])), int(np.round(fr[0][1]))

                if 0 <= fr_x < w and 0 <= fr_y < h:
                     warped_image[fr_y, fr_x] = image[y, x]
    # 填充未映射区域
    mask = np.all(warped_image == 0, axis=-1).astype(np.uint8)  # 创建掩码
    if np.any(mask):  # 如果有未填充的区域
         warped_image = cv2.inpaint(warped_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return warped_image

def run_warping(alpha_1, eps_1):
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst), alpha_1, eps_1)

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            image_output = gr.Image(label="变换结果", width=800, height=400)

    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮

    alpha_1 = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="alpha")
    eps_1 = gr.Slider(minimum=1e-8, maximum=1e-6, step=1e-8, value=1e-8, label="eps")

    inputs = [
        alpha_1, eps_1
    ]

    alpha_1.change(run_warping, inputs, image_output)
    eps_1.change(run_warping, inputs, image_output)

    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, inputs, image_output)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)


    
# 启动 Gradio 应用
demo.launch()
