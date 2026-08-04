import os
import cv2
import time
import logging
import tempfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from model import ZebraFishModel
from image_rotation_corrector import ImageRotationCorrector
from test_exact_body_region_pipeline import ExactBodyRegionAnalyzer
from config import OUTPUT_ANNOTATED_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_all_images(root_dir):
    image_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files


def process_images(custom_image_dir, base_output_dir=None):
    logger.info("Initializing models...")
    rotation_corrector = ImageRotationCorrector()
    neuron_model       = ZebraFishModel()
    analyzer           = ExactBodyRegionAnalyzer()

    if base_output_dir is None:
        base_output_dir = os.path.join(OUTPUT_ANNOTATED_PATH, "results")
    os.makedirs(base_output_dir, exist_ok=True)

    image_files = get_all_images(custom_image_dir)
    logger.info(f"Found {len(image_files)} images to process")

    summary_rows = []

    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            start_time = time.time()

            # ── Step 1: orientation correction → corrected 840×840 image + body bbox ──
            corrected_image, info = rotation_corrector.correct_image_orientation(
                image_path, save_annotated=False)

            if corrected_image is None:
                logger.warning(f"Rotation correction failed: {image_path}")
                continue

            body_bbox = None
            if info.get('corrected_landmarks') and info['corrected_landmarks'].get('body'):
                body_bbox = info['corrected_landmarks']['body']['bbox']

            if body_bbox is None:
                logger.warning(f"No body region detected: {image_path} — skipping")
                continue

            # ── Step 2: neuron detection on the corrected image ────────────────────
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, corrected_image)

            labels, boxes, confidences = neuron_model.get_predictions(tmp_path)
            os.unlink(tmp_path)

            # ── Step 3: analyze exact body region ──────────────────────────────────
            annotated_image, results = analyzer.analyze_exact_body_region(
                corrected_image, body_bbox, labels, boxes, confidences)

            # ── Step 4: save outputs ───────────────────────────────────────────────
            rel_path   = os.path.relpath(image_path, custom_image_dir)
            output_dir = os.path.join(base_output_dir, os.path.dirname(rel_path))
            os.makedirs(output_dir, exist_ok=True)

            stem = Path(image_path).stem

            # Main annotated result (neurons + spinal cord line)
            annotated_path = os.path.join(output_dir, f"annotated_{Path(image_path).name}")
            cv2.imwrite(annotated_path, annotated_image)

            # Regions image: original (left) | corrected with boxes (right)
            regions_dir = os.path.join(output_dir, "regions")
            os.makedirs(regions_dir, exist_ok=True)

            original_image = cv2.imread(image_path)
            corrected_vis  = rotation_corrector.visualize_landmarks(
                corrected_image, info.get('corrected_landmarks', {}))

            # Resize original to same height as corrected (840 px), keep aspect ratio
            h_target = corrected_vis.shape[0]
            h_orig, w_orig = original_image.shape[:2]
            w_resized = int(w_orig * h_target / h_orig)
            original_resized = cv2.resize(original_image, (w_resized, h_target))

            # Add labels
            def add_label(img, text):
                out = img.copy()
                cv2.rectangle(out, (0, 0), (out.shape[1], 28), (30, 30, 30), -1)
                cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                return out

            left  = add_label(original_resized, "Original")
            right = add_label(corrected_vis,    "Corrected (rotation)")

            comparison = np.hstack([left, right])
            regions_path = os.path.join(regions_dir, f"regions_{Path(image_path).name}")
            cv2.imwrite(regions_path, comparison)

            excel_path = os.path.join(output_dir, f"metrics_{stem}.xlsx")
            if results.get('segment_data'):
                analyzer.export_to_excel(results, excel_path)

            neuron_count = results.get('neurons_in_region', 0)
            elapsed      = time.time() - start_time
            logger.info(f"[{idx+1}/{len(image_files)}] {Path(image_path).name} "
                        f"→ {neuron_count} neurons  ({elapsed:.1f}s)")

            summary_rows.append({
                'Image':         Path(image_path).name,
                'Neurons':       neuron_count,
                'Line_px':       round(results.get('spinal_length', 0), 1),
                'Rotation_deg':  round(info.get('rotation_angle', 0), 1),
                'Annotated':     annotated_path,
                'Regions':       regions_path,
            })

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}", exc_info=True)

    # ── Summary Excel ──────────────────────────────────────────────────────────
    if summary_rows:
        summary_path = os.path.join(base_output_dir, "summary.xlsx")
        pd.DataFrame(summary_rows).to_excel(summary_path, index=False)
        logger.info(f"Summary saved: {summary_path}")

    logger.info("Pipeline completed.")


if __name__ == "__main__":
    try:
        custom_image_dir = input("Enter the path to your folder containing images: ").strip()
        if not os.path.exists(custom_image_dir):
            raise ValueError(f"Directory not found: {custom_image_dir}")

        result_dir = input("Enter the path for results folder (press Enter for default): ").strip()
        if not result_dir:
            result_dir = os.path.join(OUTPUT_ANNOTATED_PATH, "results")
        os.makedirs(result_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("DanaFish — ZEBRAFISH NEURON ANALYSIS PIPELINE")
        logger.info("1. Orientation correction (ImageRotationCorrector → blue body ROI)")
        logger.info("2. Neuron detection (YOLO26m v7, imgsz=1280, conf=0.30)")
        logger.info("3. Region analysis (ExactBodyRegionAnalyzer → spinal cord line)")
        logger.info(f"Input : {custom_image_dir}")
        logger.info(f"Output: {result_dir}")
        logger.info("=" * 60)

        process_images(custom_image_dir, result_dir)

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
