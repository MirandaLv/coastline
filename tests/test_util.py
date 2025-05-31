
import sys
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import geopandas as gpd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from util import (
    find_img_data_folder,
    re_projection,
    combine_bands,
    crop_image,
    split_and_save_patches,
)

@pytest.fixture
def dummy_tif(tmp_path):
    """Create a dummy single-band GeoTIFF file."""
    file_path = tmp_path / "band1.tif"
    data = np.ones((100, 100), dtype=np.uint16)
    transform = from_origin(0, 100, 1, 1)
    with rasterio.open(file_path, 'w', driver='GTiff', height=100, width=100, count=1,
                       dtype='uint16', crs='EPSG:32633', transform=transform) as dst:
        dst.write(data, 1)
    return file_path


@pytest.fixture
def dummy_jp2(tmp_path):
    file_path = tmp_path / "band1.jp2"
    data = np.ones((100, 100), dtype=np.uint16)
    transform = from_origin(0, 100, 1, 1)

    with rasterio.open(
            file_path, 'w',
            driver='JP2OpenJPEG',  # <- JP2 driver
            height=100,
            width=100,
            count=1,
            dtype='uint16',
            crs='EPSG:32633',
            transform=transform
    ) as dst:
        dst.write(data, 1)

    return file_path

@pytest.fixture
def dummy_multi_band_tif(tmp_path):
    """Create a dummy 3-band GeoTIFF for testing patch extraction."""
    file_path = tmp_path / "multi_band.tif"
    data = np.random.randint(0, 1000, (3, 128, 128), dtype=np.uint16)
    transform = from_origin(0, 128, 1, 1)
    with rasterio.open(file_path, 'w', driver='GTiff', height=128, width=128, count=3,
                       dtype='uint16', crs='EPSG:4326', transform=transform) as dst:
        dst.write(data)
    return file_path

@pytest.fixture
def dummy_geojson(tmp_path):
    """Create a dummy AOI GeoJSON file."""
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[box(0, 0, 50, 50)], crs='EPSG:4326')
    geojson_path = tmp_path / "aoi.geojson"
    gdf.to_file(geojson_path)
    return geojson_path

def test_re_projection(tmp_path, dummy_jp2):
    out_path = tmp_path / "reprojected.tif"
    re_projection(str(dummy_jp2), str(out_path))
    assert out_path.exists()

    with rasterio.open(out_path) as src:
        assert src.crs.to_string() == "EPSG:4326"

def test_combine_bands(tmp_path, dummy_tif):
    out_path = tmp_path / "combined.tif"
    combine_bands([str(dummy_tif), str(dummy_tif)], str(out_path))
    assert out_path.exists()

    with rasterio.open(out_path) as src:
        assert src.count == 2

def test_crop_image(tmp_path, dummy_tif, dummy_geojson):
    out_path = tmp_path / "cropped.tif"
    crop_image(str(dummy_tif), str(dummy_geojson), str(out_path))
    assert out_path.exists()

    with rasterio.open(out_path) as src:
        assert src.width < 100
        assert src.height < 100

def test_split_and_save_patches(tmp_path, dummy_multi_band_tif):
    patch_dir = tmp_path / "patches"
    split_and_save_patches(str(dummy_multi_band_tif), str(patch_dir), patch_size=64, overlap=0)

    patch_files = list(patch_dir.glob("*.tif"))
    assert len(patch_files) > 0

    with rasterio.open(patch_files[0]) as patch:
        assert patch.count == 3
        assert patch.width == 64
        assert patch.height == 64

def test_find_img_data_folder(tmp_path):
    safe_dir = tmp_path / "S2_test.SAFE" / "GRANULE" / "IMG_DATA"
    safe_dir.mkdir(parents=True)
    found = find_img_data_folder(str(tmp_path))
    assert "IMG_DATA" in found
