import json

from PIL import Image, ImageDraw

POLYGON_PATH = "data/records/20241123_121043/mask_polygon.json"



def create_mask_image(data, image_size=(1600, 600), output_file="mask_image_bw.png"):
    # Create a blank image (black background, grayscale mode)
    mask_image = Image.new("L", image_size, 0)  # "L" mode for grayscale, 0 for black
    draw = ImageDraw.Draw(mask_image)

    for item in data:
        # Extract polygon points
        polygon = [(point["x"], point["y"]) for point in item["content"]]

        # Draw the polygon on the mask (white)
        draw.polygon(polygon, fill=255)  # 255 for white

    # Save the image
    mask_image.save(output_file)
    print(f"Mask image saved to {output_file}")


if __name__ == "__main__":
    with open(POLYGON_PATH, "r") as f:
        data = json.load(f)
    create_mask_image(data)
