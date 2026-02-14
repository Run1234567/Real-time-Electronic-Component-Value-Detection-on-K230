from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os,sys,gc,time,random,utime
import ujson
from media.media import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import image
import aidemo

# 自定义YOLO检测类，继承自AIBase基类
class YOLOv12App(AIBase):
    def __init__(self, kmodel_path, model_input_size, anchors, confidence_threshold=0.5, nms_threshold=0.2, rgb888p_size=[224,224], display_size=[1920,1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)  # 调用基类的构造函数
        self.class_id = ['3.3kO','1.2R','10nF','22uF','4.7uH','330R','4.7R','100nF','10pF','4.7uF','3.3uH','200kR','1kR','0R','5.1R','2.7kR','LED','10kR','2.2kR','560pF']
        self.kmodel_path = kmodel_path  # 模型文件路径
        self.model_input_size = model_input_size  # 模型输入分辨率
        self.confidence_threshold = confidence_threshold  # 置信度阈值
        self.nms_threshold = nms_threshold  # NMS（非极大值抑制）阈值
        self.anchors = anchors  # 锚点数据，用于目标检测
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]  # sensor给到AI的图像分辨率，并对宽度进行16的对齐
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]  # 显示分辨率，并对宽度进行16的对齐
        self.debug_mode = debug_mode  # 是否开启调试模式
        self.ai2d = Ai2d(debug_mode)  # 实例化Ai2d，用于实现模型预处理
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)  # 设置Ai2d的输入输出格式和类型
        
        # 新增：用于跟踪每个类别的连续识别次数
        self.class_counter = {class_name: 0 for class_name in self.class_id}
        self.current_best_det = None  # 当前最佳检测结果

    # 配置预处理操作，这里使用了pad和resize，Ai2d支持crop/shift/pad/resize/affine，具体代码请打开/sdcard/app/libs/AI2D.py查看
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):  # 计时器，如果debug_mode大于0则开启
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size  # 初始化ai2d预处理配置，默认为sensor给到AI的尺寸，可以通过设置input_image_size自行修改输入尺寸
            top, bottom, left, right = self.get_padding_param()  # 获取padding参数
            print("padding: {} {} {} {}".format(top, bottom, left, right))
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [104, 117, 123])  # 填充边缘
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)  # 缩放图像
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])  # 构建预处理流程

    # 自定义当前任务的后处理，results是模型输出array列表
    def postprocess(self, results):
        counter = 0
        det_res = []
        with ScopedTiming("postprocess", self.debug_mode > 0):
            # 输出形状为[1, 1, 14, 2100]
            # 意思是，输出了2100个框，每个框有14个数据，其中前4个数据是xywh，后面10个是每个类别对应的置信度，这里的xy指的是中心点坐标
            for i in range(2100):
                result = results[0][0][:, i]
                max_score = max(result[4:])
                if max_score > self.confidence_threshold:
                    # 这里把位置信息恢复到1920 * 1080画布下的状态
                    x = result[0] * max(self.rgb888p_size) / max(self.model_input_size)
                    y = result[1] * max(self.rgb888p_size) / max(self.model_input_size)
                    w = result[2] * max(self.rgb888p_size) / max(self.model_input_size)
                    h = result[3] * max(self.rgb888p_size) / max(self.model_input_size)
                    class_idx = list(result[4:]).index(max_score)
                    det_res.append([x, y, w, h, class_idx, max_score])
            
            # 只取置信度最高的一个检测结果
            if det_res:
                det_res.sort(key=lambda x: x[-1], reverse=True)
                best_det = det_res[0]  # 取相似度最高的
                
                # 更新类别计数器
                current_class = self.class_id[best_det[-2]]
                for class_name in self.class_id:
                    if class_name == current_class:
                        self.class_counter[class_name] += 1
                    else:
                        self.class_counter[class_name] = 0
                
                # 只有当连续识别成功5次时才保存当前最佳检测结果
                if self.class_counter[current_class] >= 3:
                    self.current_best_det = best_det
                else:
                    self.current_best_det = None
            else:
                # 如果没有检测到任何目标，重置所有计数器
                for class_name in self.class_id:
                    self.class_counter[class_name] = 0
                self.current_best_det = None
        
        return det_res  # 返回所有检测结果用于调试，但只绘制满足条件的

    # 绘制检测结果到画面上
    def draw_result(self, pl, dets):
        with ScopedTiming("display_draw", self.debug_mode > 0):
            pl.osd_img.clear()  # 清除OSD图像
            
            # 只绘制连续识别成功5次的最佳检测结果
            if self.current_best_det is not None:
                det = self.current_best_det
                # 将检测框的坐标转换为显示分辨率下的坐标
                x, y, w, h = map(lambda x: int(round(x, 0)), det[:4])
                x = x * self.display_size[0] // self.rgb888p_size[0]
                y = y * self.display_size[1] // self.rgb888p_size[1]
                w = w * self.display_size[0] // self.rgb888p_size[0]
                h = h * self.display_size[1] // self.rgb888p_size[1]
                
                # 绘制矩形框
                pl.osd_img.draw_rectangle(x - w//2, y - h // 2, w, h, color=(255, 0, 255, 0), thickness=2)
                
                # 画标签和置信度
                class_name = self.class_id[det[-2]]
                confidence = round(det[-1], 2)
                pl.osd_img.draw_string_advanced(x - w//2, y - h // 2, 80, 
                                              "{} {}".format(class_name, confidence), 
                                              color=(255, 0, 255, 0))

    # 获取padding参数
    def get_padding_param(self):
        dst_w = self.model_input_size[0]  # 模型输入宽度
        dst_h = self.model_input_size[1]  # 模型输入高度
        ratio_w = dst_w / self.rgb888p_size[0]  # 宽度缩放比例
        ratio_h = dst_h / self.rgb888p_size[1]  # 高度缩放比例
        ratio = min(ratio_w, ratio_h)  # 取较小的缩放比例
        new_w = int(ratio * self.rgb888p_size[0])  # 新宽度
        new_h = int(ratio * self.rgb888p_size[1])  # 新高度
        dw = (dst_w - new_w) / 2  # 宽度差
        dh = (dst_h - new_h) / 2  # 高度差
        top = int(round(0))
        bottom = int(round(dh * 2 + 0.1))
        left = int(round(0))
        right = int(round(dw * 2 - 0.1))
        return top, bottom, left, right

if __name__ == "__main__":
    # 显示模式，默认"hdmi",可以选择"hdmi"和"lcd"
    display_mode="lcd"
    # k230保持不变，k230d可调整为[640,360]
    rgb888p_size = [1920, 1080]

    if display_mode=="hdmi":
        display_size=[1920,1080]
    else:
        display_size=[800,480]
    # 设置模型路径和其他参数
    kmodel_path = "/sdcard/best.kmodel"
    # 其它参数
    confidence_threshold = 0.3
    nms_threshold = 0.2
    anchor_len = None
    det_dim = 4
    anchors_path = None
    anchors = None
    anchors = None

    # 初始化PipeLine，用于图像处理流程
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create()  # 创建PipeLine实例
    # 初始化自定义人脸检测实例
    yolo_det = YOLOv12App(kmodel_path, model_input_size=[320, 320], anchors=anchors, confidence_threshold=confidence_threshold, nms_threshold=nms_threshold, rgb888p_size=rgb888p_size, display_size=display_size, debug_mode=0)
    yolo_det.config_preprocess()  # 配置预处理

    try:
        while True:
            os.exitpoint()                      # 检查是否有退出信号
            with ScopedTiming("total",1):
                img = pl.get_frame()            # 获取当前帧数据
                res = yolo_det.run(img)         # 推理当前帧
                yolo_det.draw_result(pl, res)   # 绘制结果
                pl.show_image()                 # 显示结果
                gc.collect()                    # 垃圾回收
    except Exception as e:
        print(e)                  # 打印异常信息
    finally:
        yolo_det.deinit()                       # 反初始化
        pl.destroy()                            # 销毁PipeLine实例