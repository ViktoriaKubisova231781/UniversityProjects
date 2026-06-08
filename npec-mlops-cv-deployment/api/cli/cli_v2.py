
import argparse
import requests
from pathlib import Path

def list_models():
    url = "http://localhost:8000/model_management/models"
    try:
        response = requests.get(url)
        response.raise_for_status()
        models = response.json()
        print("Available local models:")
        for model in models:
            print(f"- {model.get('id')} ({model.get('description', 'no description')})")
    except Exception as e:
        print(f"[✗] Failed to fetch models: {e}")

def segment_image(model_id, image_path=None, folder_path=None):
    url = "http://localhost:8000/segment"
    files = []
    opened = []

    if image_path:
        img = Path(image_path)
        mime = "image/tiff" if img.suffix.lower() in [".tif", ".tiff"] else "image/jpeg"
        f = open(img, 'rb')
        files.append(('files', (img.name, f, mime)))
        opened.append(f)

    elif folder_path:
        for img in Path(folder_path).glob("*.*"):
            if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                mime = "image/tiff" if img.suffix.lower() in [".tif", ".tiff"] else "image/jpeg"
                f = open(img, 'rb')
                files.append(('files', (img.name, f, mime)))
                opened.append(f)

    if not files:
        print("[✗] No valid image files found.")
        return

    data = {'model_id': model_id, 'hosting': 'local'}
    print(f"[→] Sending {len(files)} image(s) to /segment with model '{model_id}'...")

    try:
        response = requests.post(url, files=files, data=data)
    finally:
        for f in opened:
            f.close()

    if response.status_code == 200:
        content_disp = response.headers.get("Content-Disposition", "")
        filename = "output.zip"
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip("\"").strip("'")
        with open(filename, "wb") as out:
            out.write(response.content)
        print(f"[✓] Saved output to {filename}")
    else:
        print(f"[✗] Request failed: {response.status_code}")
        print(response.text)

def analyze_mask(image_path=None, folder_path=None):
    url = "http://localhost:8000/analyze_mask"
    files = []
    opened_files = []

    if image_path:
        f = open(image_path, 'rb')
        files.append(('files', (Path(image_path).name, f, 'image/tiff')))
        opened_files.append(f)
    elif folder_path:
        for img in Path(folder_path).glob("*.tif*"):
            f = open(img, 'rb')
            files.append(('files', (img.name, f, 'image/tiff')))
            opened_files.append(f)

    if not files:
        print("[✗] No valid .tif files found.")
        return

    print(f"[→] Sending {len(files)} mask(s) to /analyze_mask...")

    try:
        response = requests.post(url, files=files)
    finally:
        for f in opened_files:
            f.close()

    if response.status_code == 200:
        output_file = "analysis_output.zip"
        with open(output_file, "wb") as out:
            out.write(response.content)
        print(f"[✓] Saved analysis results to {output_file}")
    else:
        print(f"[✗] Request failed: {response.status_code}")
        print(response.text)


def analyze_image(model_id, image_path=None, folder_path=None):
    url = "http://localhost:8000/analyze_image"
    files = []
    opened_files = []

    if image_path:
        img = Path(image_path)
        mime = "image/tiff" if img.suffix.lower() in [".tif", ".tiff"] else "image/jpeg"
        f = open(img, 'rb')
        files.append(('files', (img.name, f, mime)))
        opened_files.append(f)
    elif folder_path:
        for img in Path(folder_path).glob("*.*"):
            if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                mime = "image/tiff" if img.suffix.lower() in [".tif", ".tiff"] else "image/jpeg"
                f = open(img, 'rb')
                files.append(('files', (img.name, f, mime)))
                opened_files.append(f)

    if not files:
        print("[✗] No valid image files found.")
        return

    data = {'model_id': model_id, 'hosting': 'local'}
    print(f"[→] Sending {len(files)} image(s) to /analyze_image with model '{model_id}'...")

    try:
        response = requests.post(url, files=files, data=data)
    finally:
        for f in opened_files:
            f.close()

    if response.status_code == 200:
        filename = "analyze_image_output.zip"
        content_disp = response.headers.get("Content-Disposition", "")
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip("\"").strip("'")
        with open(filename, "wb") as out:
            out.write(response.content)
        print(f"[✓] Saved output to {filename}")
    else:
        print(f"[✗] Request failed: {response.status_code}")
        print(response.text)

def full_analysis_step_one(model_id, image_path=None, folder_path=None):
    url = "http://localhost:8000/full_analysis/step_one"
    files = []
    opened_files = []

    if image_path:
        img = Path(image_path)
        mime = "image/tiff" if img.suffix.lower() in [".tif", ".tiff"] else "image/jpeg"
        f = open(img, 'rb')
        files.append(('files', (img.name, f, mime)))
        opened_files.append(f)
    elif folder_path:
        for img in Path(folder_path).glob("*.*"):
            if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                mime = "image/tiff" if img.suffix.lower() in [".tif", ".tiff"] else "image/jpeg"
                f = open(img, 'rb')
                files.append(('files', (img.name, f, mime)))
                opened_files.append(f)

    if not files:
        print("[✗] No valid image files found.")
        return

    data = {'model_id': model_id, 'hosting': 'local'}
    print(f"[→] Sending {len(files)} image(s) to /full_analysis/step_one with model '{model_id}'...")

    try:
        response = requests.post(url, files=files, data=data)
    finally:
        for f in opened_files:
            f.close()

    if response.status_code == 200:
        content_disp = response.headers.get("Content-Disposition", "")
        filename = "step_one_output.zip"
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip("\"").strip("'")
        with open(filename, "wb") as out:
            out.write(response.content)
        print(f"[✓] Saved output to {filename}")
    else:
        print(f"[✗] Request failed: {response.status_code}")
        print(response.text)

def full_analysis_step_two(image_path=None, folder_path=None):
    url = "http://localhost:8000/full_analysis/step_two"
    files = []
    opened = []

    if image_path:
        f = open(image_path, 'rb')
        files.append(('files', (Path(image_path).name, f, 'image/tiff')))
        opened.append(f)
    elif folder_path:
        for img in Path(folder_path).glob("*.tif*"):
            f = open(img, 'rb')
            files.append(('files', (img.name, f, 'image/tiff')))
            opened.append(f)

    if not files:
        print("[✗] No valid .tif files found.")
        return

    print(f"[→] Sending {len(files)} mask(s) to /full_analysis/step_two...")

    try:
        response = requests.post(url, files=files)
    finally:
        for f in opened:
            f.close()

    if response.status_code == 200:
        filename = "step_two_results.zip"
        content_disp = response.headers.get("Content-Disposition", "")
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip("\"").strip("'")
        with open(filename, "wb") as out:
            out.write(response.content)
        print(f"[✓] Saved output to {filename}")
    else:
        print(f"[✗] Request failed: {response.status_code}")
        print(response.text)


def full_analysis_step_three(model_id, image_path=None, folder_path=None):
    url = "http://localhost:8000/full_analysis/step_three"
    files = []
    opened = []

    if image_path:
        img = Path(image_path)
        mime = "image/tiff" if img.suffix.lower() in [".tif", ".tiff"] else "image/jpeg"
        f = open(img, 'rb')
        files.append(('files', (img.name, f, mime)))
        opened.append(f)
    elif folder_path:
        for img in Path(folder_path).glob("*.*"):
            if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                mime = "image/tiff" if img.suffix.lower() in [".tif", ".tiff"] else "image/jpeg"
                f = open(img, 'rb')
                files.append(('files', (img.name, f, mime)))
                opened.append(f)

    if not files:
        print("[✗] No valid image files found.")
        return

    data = {'model_id': model_id, 'hosting': 'local'}
    print(f"[→] Sending {len(files)} image(s) to /full_analysis/step_three with model '{model_id}'...")

    try:
        response = requests.post(url, files=files, data=data)
    finally:
        for f in opened:
            f.close()

    if response.status_code == 200:
        content_disp = response.headers.get("Content-Disposition", "")
        filename = "step_three_overlays.zip"
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip("\"").strip("'")
        with open(filename, "wb") as out:
            out.write(response.content)
        print(f"[✓] Saved overlay results to {filename}")
    else:
        print(f"[✗] Request failed: {response.status_code}")
        print(response.text)


def full_analysis_step_four(csv_path):
    url = "http://localhost:8000/full_analysis/step_four"
    if not Path(csv_path).exists():
        print(f"[✗] CSV file not found: {csv_path}")
        return

    with open(csv_path, "rb") as f:
        files = {"csv_file": (Path(csv_path).name, f, "text/csv")}
        print(f"[→] Sending CSV to /full_analysis/step_four...")

        response = requests.post(url, files=files)

        if response.status_code == 200:
            output_file = "root_analysis_report.pdf"
            content_disp = response.headers.get("Content-Disposition", "")
            if "filename=" in content_disp:
                output_file = content_disp.split("filename=")[-1].strip("\"").strip("'")
            with open(output_file, "wb") as out:
                out.write(response.content)
            print(f"[✓] Saved report to {output_file}")
        else:
            print(f"[✗] Request failed: {response.status_code}")
            print(response.text)



def main():
    parser = argparse.ArgumentParser(description="CLI for segmentation & model listing")
    subparsers = parser.add_subparsers(dest="command")

    # List models
    list_parser = subparsers.add_parser("list", help="List local models")

    # Segment images
    segment_parser = subparsers.add_parser("segment", help="Run segmentation with a local model")
    segment_parser.add_argument("--model_id", required=True, help="ID of the local model")
    segment_parser.add_argument("--image", help="Path to a single image")
    segment_parser.add_argument("--folder", help="Path to a folder of images")

    # Analyze mask
    analyze_parser = subparsers.add_parser("analyze_mask", help="Analyze masks to extract root measurements")
    analyze_parser.add_argument("--image", help="Single .tif mask")
    analyze_parser.add_argument("--folder", help="Folder of .tif masks")

    # Analyze image
    image_analysis_parser = subparsers.add_parser("analyze_image", help="Run full image analysis with a local model")
    image_analysis_parser.add_argument("--model_id", required=True, help="ID of the local model")
    image_analysis_parser.add_argument("--image", help="Path to a single image")
    image_analysis_parser.add_argument("--folder", help="Path to a folder of images")

    # Full analysis step one
    step_one_parser = subparsers.add_parser("full_analysis_step_one", help="Run full analysis step one with a local model")
    step_one_parser.add_argument("--model_id", required=True, help="ID of the local model")
    step_one_parser.add_argument("--image", help="Path to a single image")
    step_one_parser.add_argument("--folder", help="Path to a folder of images")

    # Full analysis step two
    step_two_parser = subparsers.add_parser("full_analysis_step_two", help="Run full analysis step two on mask(s)")
    step_two_parser.add_argument("--image", help="Path to a single .tif mask")
    step_two_parser.add_argument("--folder", help="Path to a folder of .tif masks")

    # Full analysis step three
    step_three_parser = subparsers.add_parser("full_analysis_step_three", help="Run full analysis step three (image overlays)")
    step_three_parser.add_argument("--model_id", required=True, help="ID of the local model")
    step_three_parser.add_argument("--image", help="Path to a single image")
    step_three_parser.add_argument("--folder", help="Path to a folder of images")

    # Full analysis step four
    step_four_parser = subparsers.add_parser("full_analysis_step_four", help="Generate root analysis PDF from a CSV")
    step_four_parser.add_argument("--csv", required=True, help="Path to root measurements CSV")


    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "segment":
        if args.image:
            segment_image(args.model_id, image_path=args.image)
        elif args.folder:
            segment_image(args.model_id, folder_path=args.folder)
        else:
            print("[✗] You must provide either --image or --folder")
    elif args.command == "analyze_mask":
        if args.image:
            analyze_mask(image_path=args.image)
        elif args.folder:
            analyze_mask(folder_path=args.folder)
        else:
            print("[✗] You must provide either --image or --folder")
    elif args.command == "analyze_image":
        if args.image:
            analyze_image(args.model_id, image_path=args.image)
        elif args.folder:
            analyze_image(args.model_id, folder_path=args.folder)
        else:
            print("[✗] You must provide either --image or --folder")
    elif args.command == "full_analysis_step_one":
        if args.image:
            full_analysis_step_one(args.model_id, image_path=args.image)
        elif args.folder:
            full_analysis_step_one(args.model_id, folder_path=args.folder)
        else:
            print("[✗] You must provide either --image or --folder")
    elif args.command == "full_analysis_step_two":
        if args.image:
            full_analysis_step_two(image_path=args.image)
        elif args.folder:
            full_analysis_step_two(folder_path=args.folder)
        else:
            print("[✗] You must provide either --image or --folder")
    elif args.command == "full_analysis_step_three":
        if args.image:
            full_analysis_step_three(args.model_id, image_path=args.image)
        elif args.folder:
            full_analysis_step_three(args.model_id, folder_path=args.folder)
        else:
            print("[✗] You must provide either --image or --folder")
    elif args.command == "full_analysis_step_four":
        full_analysis_step_four(args.csv)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
