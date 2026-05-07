# setup_dataset.py
# This script automatically creates test folder
# by moving some images from train to test

import os
import shutil
import random

random.seed(42)  # So same images selected every time


def setup_test_folders():
    """Create test folders and move images there"""

    print("="*55)
    print("  SETTING UP DATASET FOLDERS")
    print("="*55)

    # Create test folders
    os.makedirs('dataset/test/normal', exist_ok=True)
    os.makedirs('dataset/test/cancer', exist_ok=True)
    print("\n✅ Created test folders")

    # Process each class
    for class_name in ['normal', 'cancer']:

        train_path = f'dataset/train/{class_name}'
        test_path  = f'dataset/test/{class_name}'

        # Get all image files
        all_files = [
            f for f in os.listdir(train_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        if len(all_files) == 0:
            print(f"❌ No images found in {train_path}")
            continue

        print(f"\n📁 {class_name}:")
        print(f"   Total images found : {len(all_files)}")

        # Shuffle randomly
        random.shuffle(all_files)

        # Take 20% for testing
        test_count  = max(1, int(len(all_files) * 0.20))
        test_files  = all_files[:test_count]
        train_files = all_files[test_count:]

        print(f"   Moving to test     : {test_count} images")
        print(f"   Keeping in train   : {len(train_files)} images")

        # Move files to test folder
        moved = 0
        for filename in test_files:
            src = os.path.join(train_path, filename)
            dst = os.path.join(test_path,  filename)

            # Make sure destination doesn't already exist
            if not os.path.exists(dst):
                shutil.move(src, dst)
                moved += 1

        print(f"   ✅ Successfully moved: {moved} images")

    # Final count
    print("\n" + "="*55)
    print("  FINAL DATASET COUNT")
    print("="*55)

    total_train = 0
    total_test  = 0

    for split in ['train', 'test']:
        for cls in ['normal', 'cancer']:
            path = f'dataset/{split}/{cls}'
            if os.path.exists(path):
                files = [
                    f for f in os.listdir(path)
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
                ]
                count = len(files)
                print(f"  {split:5s}/{cls:8s} : {count:4d} images")

                if split == 'train':
                    total_train += count
                else:
                    total_test += count
            else:
                print(f"  {split}/{cls}: ❌ NOT FOUND")

    print(f"\n  Total Train : {total_train}")
    print(f"  Total Test  : {total_test}")
    print(f"  Grand Total : {total_train + total_test}")
    print("="*55)
    print("\n✅ Dataset setup complete!")
    print("Now run: python train_model.py")


if __name__ == "__main__":
    setup_test_folders()
