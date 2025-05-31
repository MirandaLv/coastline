
import os
import rasterio
from pathlib import Path
from util import find_img_data_folder, re_projection, combine_bands, crop_image, split_and_save_patches
import glob


base_path = Path(os.getcwd())
raw_data_path = os.path.join(base_path, "Sentinel-2")
aoi_path = os.path.join(base_path, "boundary.geojson")

b_path = os.path.join(base_path, "data/sentinel")

# Setting up different directories for the data
(base_path / 'data' / 'sentinel').mkdir(exist_ok=True, parents=True)
(base_path / 'data' /'sentinel' / 'reprojection').mkdir(exist_ok=True, parents=True)
reproject_path = os.path.join(b_path, 'reprojection')

merge_output_path = os.path.join(b_path, "combined_4band.tif")
# crop_aoi_path = os.path.join(b_path, "aoi_imagery.tif")
output_dir = os.path.join(b_path, "patches")

band_list = ["02", "03", "04", "08"] # processing the 10m bands
all_bands_path = []

for b in band_list:

    print("Working on band {}".format(b))

    img_data_path = find_img_data_folder(raw_data_path)

    band_granules_path = glob.glob(
        os.path.join(img_data_path, 'R10m/*_B{}_*.jp2'.format(b)))

    band_granules_src = [rasterio.open(i, driver='JP2OpenJPEG') for i in band_granules_path]
    proj_band_path = [os.path.join(reproject_path, Path(i).stem + '_wgs84.tif') for i in band_granules_path]

    all_bands_path.append(proj_band_path[0])

    # Reproject all bands to wgs-84
    for file, outfile in zip(band_granules_path, proj_band_path):

        infilename = os.path.basename(file).split('.')[0]
        outfilename_pre = os.path.basename(outfile).split('.')[0][0:-6]

        if infilename == outfilename_pre and not os.path.isfile(outfile):
            print("Reproject file {}".format(file))

            re_projection(file, outfile)


combine_bands(all_bands_path, merge_output_path)
print("merged_imagery_path is {}".format(merge_output_path))

# crop_image(merge_output_path, aoi_path, crop_aoi_path)
split_and_save_patches(merge_output_path, output_dir, patch_size=512, overlap=10, bands=[1, 2, 3, 4], skip_partial=True)