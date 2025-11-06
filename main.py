import os
import sys
import json
import argparse
from typing import Tuple, Optional, Any, Dict

# Allow importing the local package without installation
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.join(CURRENT_DIR, "retinaface")
if PACKAGE_ROOT not in sys.path:
	sys.path.insert(0, PACKAGE_ROOT)

from retinaface import RetinaFace  # type: ignore


def parse_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
	if not size_str:
		return None
	parts = size_str.lower().split("x")
	if len(parts) != 2:
		raise argparse.ArgumentTypeError("--target-size 需要形如 224x224")
	w, h = int(parts[0]), int(parts[1])
	return (w, h)


def _to_serializable(obj: Any) -> Any:
	# 处理 numpy 标量与数组，转换为原生 Python 类型，便于 json 序列化
	try:
		import numpy as np  # pylint: disable=import-error
		if isinstance(obj, np.generic):
			return obj.item()
		if isinstance(obj, np.ndarray):
			return obj.tolist()
	except Exception:
		pass
	return obj


def _draw_detections(img_bgr, detections: Dict[str, Any]):
	import cv2  # pylint: disable=import-error
	for _, face in detections.items():
		x1, y1, x2, y2 = face["facial_area"]
		cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
		lm = face.get("landmarks", {})
		for k in ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]:
			pt = lm.get(k)
			if isinstance(pt, (list, tuple)) and len(pt) == 2:
				x, y = int(pt[0]), int(pt[1])
				cv2.circle(img_bgr, (x, y), 2, (0, 0, 255), -1)
		score = face.get("score")
		if score is not None:
			cv2.putText(img_bgr, f"{score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
	return img_bgr


def cmd_detect(args: argparse.Namespace) -> None:
	resp = RetinaFace.detect_faces(
		img_path=args.image,
		threshold=args.threshold,
		model=None,
		allow_upscaling=not args.no_upscale,
	)
	print(json.dumps(resp, ensure_ascii=False, indent=2, default=_to_serializable))


def cmd_extract(args: argparse.Namespace) -> None:
	faces = RetinaFace.extract_faces(
		img_path=args.image,
		threshold=args.threshold,
		model=None,
		align=not args.no_align,
		allow_upscaling=not args.no_upscale,
		expand_face_area=args.expand,
		target_size=parse_size(args.target_size),
		min_max_norm=args.min_max_norm,
	)
	os.makedirs(args.output, exist_ok=True)
	for idx, face in enumerate(faces, start=1):
		# face is RGB numpy array in HxWxC
		out_path = os.path.join(args.output, f"face_{idx}.png")
		# 处理归一化后的图像：如果值在 [0,1] 且为 float，转成 uint8 [0,255]
		import numpy as np  # pylint: disable=import-error
		if face.dtype == np.float32 or face.dtype == np.float64:
			if face.max() <= 1.0:
				face = (face * 255).astype(np.uint8)
		if face.dtype != np.uint8:
			face = face.astype(np.uint8)
		# save with cv2 (expects BGR) if available, else fallback to PIL
		try:
			import cv2  # pylint: disable=import-error
			cv2.imwrite(out_path, face[:, :, ::-1])
		except Exception:  # noqa: BLE001 - fallback to PIL
			from PIL import Image  # pylint: disable=import-error
			Image.fromarray(face).save(out_path)
		print(f"saved: {out_path}")
	print(f"total faces: {len(faces)}")


def cmd_visualize(args: argparse.Namespace) -> None:
	# 读取原图(BGR)，检测后画框并保存/展示
	import cv2  # pylint: disable=import-error
	img = cv2.imread(args.image)
	if img is None:
		raise FileNotFoundError(f"无法读取图片: {args.image}")
	resp = RetinaFace.detect_faces(
		img_path=img[:, :, ::-1],  # 输入为 RGB 或路径，这里转换为 RGB 以复用预处理
		threshold=args.threshold,
		model=None,
		allow_upscaling=not args.no_upscale,
	)
	img = _draw_detections(img, resp)
	if args.show:
		cv2.imshow("retinaface-visualize", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	if args.output:
		os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
		cv2.imwrite(args.output, img)
		print(f"saved: {args.output}")


def cmd_webcam(args: argparse.Namespace) -> None:
	import cv2  # pylint: disable=import-error
	cap = cv2.VideoCapture(args.camera)
	if args.width:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
	if args.height:
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
	print("按 q 退出窗口")
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		resp = RetinaFace.detect_faces(
			img_path=frame[:, :, ::-1],
			threshold=args.threshold,
			model=None,
			allow_upscaling=not args.no_upscale,
		)
		frame = _draw_detections(frame, resp)
		cv2.imshow("retinaface-webcam", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	cap.release()
	cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="retinaface-demo",
		description="RetinaFace 示例：人脸检测、对齐提取、图片可视化与摄像头实时",
	)
	sub = p.add_subparsers(dest="command", required=True)

	# detect
	pd = sub.add_parser("detect", help="检测人脸并输出JSON")
	pd.add_argument("--image", required=True, help="输入图片路径")
	pd.add_argument("--threshold", type=float, default=0.9, help="检测阈值，默认0.9")
	pd.add_argument("--no-upscale", action="store_true", help="禁止小图上采样")
	pd.set_defaults(func=cmd_detect)

	# extract
	pe = sub.add_parser("extract", help="检测+对齐并裁剪保存人脸")
	pe.add_argument("--image", required=True, help="输入图片路径")
	pe.add_argument("--output", default="outputs", help="输出目录，默认outputs")
	pe.add_argument("--threshold", type=float, default=0.9, help="检测阈值，默认0.9")
	pe.add_argument("--no-align", action="store_true", help="不做对齐")
	pe.add_argument("--no-upscale", action="store_true", help="禁止小图上采样")
	pe.add_argument("--expand", type=int, default=0, help="扩大人脸区域百分比")
	pe.add_argument("--target-size", default=None, help="目标尺寸，如 224x224")
	pe.add_argument("--min-max-norm", action="store_true", help="与 --target-size 搭配：归一化到[0,1]")
	pe.set_defaults(func=cmd_extract)

	# visualize
	pv = sub.add_parser("visualize", help="在图片上画框/关键点并保存或预览")
	pv.add_argument("--image", required=True, help="输入图片路径")
	pv.add_argument("--output", default="visualized.jpg", help="输出图片路径")
	pv.add_argument("--threshold", type=float, default=0.9, help="检测阈值，默认0.9")
	pv.add_argument("--no-upscale", action="store_true", help="禁止小图上采样")
	pv.add_argument("--show", action="store_true", help="弹窗显示结果")
	pv.set_defaults(func=cmd_visualize)

	# webcam
	pw = sub.add_parser("webcam", help="摄像头实时检测(按 q 退出)")
	pw.add_argument("--camera", type=int, default=0, help="摄像头索引，默认0")
	pw.add_argument("--threshold", type=float, default=0.9, help="检测阈值，默认0.9")
	pw.add_argument("--no-upscale", action="store_true", help="禁止小图上采样")
	pw.add_argument("--width", type=int, default=640, help="捕获宽度")
	pw.add_argument("--height", type=int, default=480, help="捕获高度")
	pw.set_defaults(func=cmd_webcam)

	return p


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
