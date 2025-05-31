

import os
import argparse
import torch
from dataloader import SentinelDataset
from torch.utils.data import DataLoader

from inference import run_inference
from stitching import stitch_tiff_patches
from model_utils import load_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.weights, device=device)
    dataset = SentinelDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    output_dir = os.path.join(args.output_dir, "inference_outputs")
    # run_inference(model, dataloader, output_dir, device)

    if args.stitch:
        print("Stitching??")
        output_mosaic_path = os.path.join(args.output_dir, "stitched_output.tif")
        stitch_tiff_patches(output_dir, output_mosaic_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference + Stitching pipeline for segmentation models on GeoTIFF patches.")
    parser.add_argument("--data_path", required=True, help="Path to folder with input TIFF patches.")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth).")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save inference and stitched outputs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--stitch", action="store_false", help="Stitch output TIFFs into a mosaic.")

    args = parser.parse_args()
    main(args)



