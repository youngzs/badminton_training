# -*- coding: utf-8 -*-
"""
诊断已生成的视频文件
"""
import sys
import cv2
import numpy as np

# 设置Windows控制台输出编码
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


def analyze_video(video_path):
    """分析视频文件"""

    print(f"分析视频: {video_path}")
    print("=" * 60)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    print(f"分辨率: {width} x {height}")
    print(f"帧率: {fps} FPS")
    print(f"总帧数: {total_frames}")
    print(f"编码器: {fourcc}")

    # 检查音频
    print(f"\n音频检查:")
    print(f"  OpenCV不直接支持音频检查")
    print(f"  建议使用媒体播放器验证音频")

    # 读取并分析几帧
    print(f"\n帧分析:")

    frame_indices = [0, total_frames // 4, total_frames // 2, total_frames * 3 // 4, total_frames - 1]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            # 计算帧的颜色统计
            mean_color = np.mean(frame, axis=(0, 1))
            std_color = np.std(frame, axis=(0, 1))

            # 检测是否有彩色线条（可能是pose标注）
            # 查找红色、绿色、蓝色、黄色的像素
            red_pixels = np.sum((frame[:, :, 2] > 200) & (frame[:, :, 1] < 100) & (frame[:, :, 0] < 100))
            green_pixels = np.sum((frame[:, :, 1] > 200) & (frame[:, :, 2] < 100) & (frame[:, :, 0] < 100))
            blue_pixels = np.sum((frame[:, :, 0] > 200) & (frame[:, :, 1] < 100) & (frame[:, :, 2] < 100))
            yellow_pixels = np.sum((frame[:, :, 1] > 200) & (frame[:, :, 2] > 200) & (frame[:, :, 0] < 100))

            has_pose = (red_pixels + green_pixels + blue_pixels + yellow_pixels) > 100

            print(f"  帧 {idx:5d}: 均值={mean_color.astype(int)} 标准差={std_color.astype(int)} "
                  f"彩色像素={red_pixels + green_pixels + blue_pixels + yellow_pixels:6d} "
                  f"{'✓ 可能有pose' if has_pose else '✗ 无pose'}")

    cap.release()

    print("\n" + "=" * 60)
    print("分析完成！")


def compare_videos(original_path, processed_path):
    """对比两个视频"""

    print("对比两个视频文件")
    print("=" * 60)

    cap1 = cv2.VideoCapture(original_path)
    cap2 = cv2.VideoCapture(processed_path)

    if not cap1.isOpened():
        print(f"错误: 无法打开原始视频")
        return

    if not cap2.isOpened():
        print(f"错误: 无法打开处理视频")
        cap1.release()
        return

    # 读取第一帧对比
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        # 计算差异
        diff = cv2.absdiff(frame1, frame2)
        diff_sum = np.sum(diff)

        print(f"原始视频第一帧: {frame1.shape}")
        print(f"处理视频第一帧: {frame2.shape}")
        print(f"帧差异总和: {diff_sum}")

        if diff_sum == 0:
            print("⚠ 警告: 两个视频第一帧完全相同！")
            print("   可能问题：")
            print("   1. 处理时没有成功绘制pose标注")
            print("   2. 加载了错误的文件")
        elif diff_sum < 1000000:
            print("⚠ 警告: 两个视频第一帧几乎相同（差异很小）")
            print("   可能原因：pose标注未成功绘制")
        else:
            print("✓ 两个视频第一帧有明显差异")

    cap1.release()
    cap2.release()

    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"\n单个视频分析")
        analyze_video(video_path)
    else:
        # 默认测试路径
        original = "F:/videos/ytDownload/input/demo2.mp4"
        processed = "F:/videos/ytDownload/input/demo2_analyzed.mp4"

        print(f"\n原始视频分析")
        analyze_video(original)

        print(f"\n\n处理视频分析")
        analyze_video(processed)

        print(f"\n\n对比分析")
        compare_videos(original, processed)
