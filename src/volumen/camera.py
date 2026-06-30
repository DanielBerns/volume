from PIL import Image
from PIL.ExifTags import TAGS
from typing import Dict, Any, Optional

def extract_exif_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts EXIF metadata from an image, focusing on camera intrinsic parameters.
    """
    metadata = {}
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    # We are particularly interested in FocalLength
                    if tag_name in ['FocalLength', 'FocalLengthIn35mmFilm', 'Make', 'Model']:
                        metadata[tag_name] = str(value)
    except Exception as e:
        print(f"Failed to extract EXIF from {image_path}: {e}")
    
    return metadata

def get_focal_length(metadata: Dict[str, Any], default: float = 50.0) -> float:
    """
    Attempts to parse the focal length from metadata.
    Returns a default value if not found or parsing fails.
    """
    if 'FocalLengthIn35mmFilm' in metadata:
        try:
            return float(metadata['FocalLengthIn35mmFilm'])
        except ValueError:
            pass
            
    if 'FocalLength' in metadata:
        try:
            val = metadata['FocalLength']
            # Pillow sometimes returns IFDRational for FocalLength, which evaluates to a float or a tuple.
            if isinstance(val, tuple) and len(val) == 2:
                return float(val[0]) / float(val[1])
            return float(val)
        except (ValueError, TypeError, ZeroDivisionError):
            pass
            
    return default
