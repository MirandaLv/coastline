
import os
import rasterio
from rasterio.merge import merge
import argparse

def stitch_tiff_patches(input_dir, output_path):
    """Stitch GeoTIFF patches into a single mosaic."""
    # Find all .tif files in the directory
    tif_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".tif")
    ]

    # Open all the datasets
    src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]

    mosaic, out_transform = merge(src_files_to_mosaic)

    # Use metadata from the first file as a template
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    # Write the stitched image
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

    print(f"Stitched image saved to: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch multiple GeoTIFF patches into a single mosaic.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing GeoTIFF patches.")
    parser.add_argument("output_path", type=str, help="Output path for the stitched GeoTIFF.")

    args = parser.parse_args()

    stitch_tiff_patches(args.input_dir, args.output_path)
