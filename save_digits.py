import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

def save_sample_digits():
    # Load dataset
    digits = load_digits()
    
    # Create output directory
    output_dir = "sample_digits"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Track which digits we've found
    saved_digits = set()
    
    # Iterate through images and save the first occurrence of each digit
    for image_data, target in zip(digits.images, digits.target):
        if target not in saved_digits:
            filename = os.path.join(output_dir, f"digit_{target}.png")
            
            # Save the image
            # cmap='gray_r' to invert (black text on white background) which is typical for saved images
            # But wait, our training data is white on black.
            # If we save using matplotlib's default savefig, it adds whitespace padding and axes.
            # We want just the raw pixels if possible, or a clean plot.
            
            # Let's clean it up to be just the digit.
            plt.figure(figsize=(2, 2))
            plt.imshow(image_data, cmap='gray', interpolation='nearest') # grayscale (white on black)
            plt.axis('off')
            
            # To save exactly the pixel data is tricky with matplotlib padding.
            # However, for user testing, a standard matplotlib save is usually fine visually.
            # BUT, the user wants "number in the pixel formate". 
            # If they want to upload it to the app, the app expects an image.
            # Let's save it as a straightforward image.
            
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            print(f"Saved {filename}")
            saved_digits.add(target)
            
        if len(saved_digits) == 10:
            break

if __name__ == "__main__":
    save_sample_digits()
