import os
from PIL import Image, ImageDraw, ImageFont

def create_favicon(output_path, sizes=[16, 32, 48, 64]):
    """
    Create a simple favicon with a soundwave-like pattern using PIL
    """
    images = []
    
    for size in sizes:
        # Create a transparent background image
        img = Image.new('RGBA', (size, size), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Calculate positioning
        padding = int(size * 0.1)
        width = size - 2 * padding
        height = size - 2 * padding
        
        # Background blue square with rounded corners (approximate)
        draw.rectangle(
            [(padding, padding), (size - padding, size - padding)],
            fill=(13, 110, 253),  # Bootstrap primary blue
            width=0,
        )
        
        # Draw a simple soundwave-like pattern
        wave_color = (255, 255, 255)  # White
        center_y = size // 2
        line_width = max(1, int(size * 0.06))
        
        # Center line
        draw.line(
            [(padding + width * 0.2, center_y), (padding + width * 0.8, center_y)],
            fill=wave_color,
            width=line_width
        )
        
        # Upper waves
        offset = int(height * 0.15)
        draw.line(
            [(padding + width * 0.3, center_y - offset), 
             (padding + width * 0.7, center_y - offset)],
            fill=wave_color,
            width=line_width
        )
        
        draw.line(
            [(padding + width * 0.4, center_y - offset * 2), 
             (padding + width * 0.6, center_y - offset * 2)],
            fill=wave_color,
            width=line_width
        )
        
        # Lower waves
        draw.line(
            [(padding + width * 0.3, center_y + offset), 
             (padding + width * 0.7, center_y + offset)],
            fill=wave_color,
            width=line_width
        )
        
        draw.line(
            [(padding + width * 0.4, center_y + offset * 2), 
             (padding + width * 0.6, center_y + offset * 2)],
            fill=wave_color,
            width=line_width
        )
        
        images.append(img)
    
    # Save as ICO file with multiple sizes
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images[0].save(
        output_path,
        format='ICO', 
        sizes=[(img.width, img.height) for img in images],
        append_images=images[1:]
    )
    print(f"Favicon created at {output_path}")

# Create directories if they don't exist
favicon_path = '../static/favicon.ico'

# Create the favicon
create_favicon(favicon_path) 