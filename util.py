
# Extract the Sentinel 2 data by bands and merge into one imagery

import os
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
from rasterio.mask import mask
from rasterio.windows import Window
from tqdm import tqdm


def find_img_data_folder(root_path):
    """
    Searching for the correct directory that has the raw Sentinel data downloaded

    :param root_path: Path to sentinel data folder
    :return:
    """
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Look for folders that end with '.SAFE'
        for dirname in dirnames:
            if dirname.endswith('.SAFE'):
                safe_path = os.path.join(dirpath, dirname)
                # Now search inside the .SAFE directory for IMG_DATA
                for sub_dirpath, sub_dirnames, _ in os.walk(safe_path):
                    if "IMG_DATA" in sub_dirnames:
                        return os.path.join(sub_dirpath, "IMG_DATA")
    return None  # Not found


def re_projection(intif, outtif):
    """
    Image reprojection to project the raw imagery bands into WGS-84

    :param intif: input imagery path
    :param outtif: reprojected imagery
    :return:
    """
    dst_crs = 'EPSG:4326'

    src = rasterio.open(intif, driver='JP2OpenJPEG')
    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height,
        'driver': 'GTiff'
    })

    with rasterio.open(outtif, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)


def combine_bands(band_paths, output_path):
    """
    The raw imagery are downloaded as individual band, merge the band into a multisplectral imagery
    :param band_paths: path to all band info
    :param output_path: output path
    :return:
    """
    # Open all the input bands
    band_data = [rasterio.open(path) for path in band_paths]

    # Check if all input bands have the same dimensions and CRS
    ref_profile = band_data[0].profile
    for b in band_data:
        assert b.width == ref_profile['width'] and b.height == ref_profile['height'], "Band dimensions do not match"
        assert b.crs == ref_profile['crs'], "Band CRS do not match"
        assert b.transform == ref_profile['transform'], "Band transforms do not match"

    # Update the profile to write multiple bands
    out_profile = ref_profile.copy()
    out_profile.update(count=len(band_data))

    # Write to output file
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        for i, src in enumerate(band_data, start=1):
            dst.write(src.read(1), i)

    # Close all band files
    for src in band_data:
        src.close()

    print(f"Combined TIFF written to: {output_path}")


def crop_image(tiff_path, aoi_path, output_path):
    """
    Crop the raw imagery by AOI boundary

    :param tiff_path: raw imagery tiff
    :param aoi_path: AOI file path
    :param output_path: output path
    :return:
    """
    with fiona.open(aoi_path, "r") as shapefile:
        aoi_geometries = [feature["geometry"] for feature in shapefile]

    with rasterio.open(tiff_path) as src:
        # Crop the image using the AOI geometry
        cropped_image, cropped_transform = mask(src, aoi_geometries, crop=True)

        # Update the metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "height": cropped_image.shape[1],
            "width": cropped_image.shape[2],
            "transform": cropped_transform
        })

    # Write the cropped image to a new file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(cropped_image)

    print(f"Cropped image saved to: {output_path}")



def split_and_save_patches(input_tif, output_dir, patch_size=256, overlap=0, bands=None, skip_partial=True):
    """
    Create image patches for model training/inferencing

    Parameters:
        input_tif (str): Path to input image.
        output_dir (str): Output directory for patches.
        patch_size (int): Patch size in pixels (square).
        overlap (int): Overlap between patches in pixels.
        bands (list or None): List of band indices to include (1-based). If None, includes all.
        skip_partial (bool): Skip patches that would go beyond image bounds.
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_tif) as src:
        img_width = src.width
        img_height = src.height
        total_bands = src.count if bands is None else len(bands)

        step = patch_size - overlap
        count = 0

        col_steps = range(0, img_width - (patch_size if skip_partial else 0) + 1, step)
        row_steps = range(0, img_height - (patch_size if skip_partial else 0) + 1, step)

        for top in tqdm(row_steps, desc="Processing rows"):
            for left in col_steps:
                win_width = min(patch_size, img_width - left)
                win_height = min(patch_size, img_height - top)

                if skip_partial and (win_width < patch_size or win_height < patch_size):
                    continue

                window = Window(left, top, win_width, win_height)
                transform = src.window_transform(window)

                # Read only selected bands and window
                patch = src.read(indexes=bands, window=window) if bands else src.read(window=window)

                meta = src.meta.copy()
                meta.update({
                    "driver": "GTiff",
                    "height": win_height,
                    "width": win_width,
                    "transform": transform,
                    "count": total_bands
                })

                patch_path = os.path.join(output_dir, f"patch_{count:05d}.tif")
                with rasterio.open(patch_path, 'w', **meta) as dst:
                    dst.write(patch)

                count += 1

    print(f"{count} patches saved to {output_dir}")


