import re

def extract_hole_area(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    in_external = False
    rectangles = []
    current_coords = []

    for line in lines:
        line = line.strip()

        if ";TYPE:External perimeter" in line:
            if current_coords:
                rectangles.append(current_coords)
            current_coords = []
            in_external = True
            continue

        if line.startswith(";TYPE:") and "External" not in line:
            if current_coords:
                rectangles.append(current_coords)
                current_coords = []
            in_external = False
            continue

        if not in_external:
            continue

        if "F18000" in line:  # skip travel moves
            continue

        m = re.match(r"G1\s+X([\d.]+)\s+Y([\d.]+)", line)
        if m:
            current_coords.append((float(m.group(1)), float(m.group(2))))

    if current_coords:
        rectangles.append(current_coords)

    seen = set()
    for coords in rectangles:
        if len(coords) < 3:
            continue
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        width  = round(max(xs) - min(xs), 4)
        height = round(max(ys) - min(ys), 4)

        # Outer border will be ~19-20mm, inner holes ~3-4mm
        # Only keep shapes with both dimensions under 10mm
        if width > 10 or height > 10:
            continue
        if width < 1 or height < 1:  # skip degenerate segments
            continue

        key = (width, height)
        if key not in seen:
            seen.add(key)
            # Subtract one wall width (0.45mm) per side for clear inner area
            clear_w = round(width  - 2 * 0.45, 4)
            clear_h = round(height - 2 * 0.45, 4)
            area_centerline = round(width * height, 4)
            area_clear      = round(clear_w * clear_h, 4)
            print(f"Hole (centerline): {width} x {height} mm  →  {area_centerline} mm²")
            print(f"Hole (clear area): {clear_w} x {clear_h} mm  →  {area_clear} mm²")

extract_hole_area("part_files/PluginTest2.gcode")