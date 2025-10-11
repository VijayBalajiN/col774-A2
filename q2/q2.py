import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from svm import SupportVectorMachine
import time

def load_images_from_folder(folder_path, target_size=(32, 32)):
    """Load and preprocess images from a folder"""
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img = img.resize(target_size)
            img_array = np.array(img)
            
            # Ensure RGB format
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            
            # Flatten and normalize
            img_flat = img_array.flatten() / 255.0
            images.append(img_flat)
    
    return np.array(images)

def load_binary_data(data_dir, class1_idx, class2_idx):
    """Load binary classification data for two classes"""
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(classes)
    print(class1_idx, class2_idx)
    class1_name = classes[class1_idx]
    class2_name = classes[class2_idx]
    
    print(f"Loading classes: {class1_name} (0) vs {class2_name} (1)")
    
    # Load training data
    train_class1 = load_images_from_folder(os.path.join(data_dir, 'train', class1_name))
    train_class2 = load_images_from_folder(os.path.join(data_dir, 'train', class2_name))
    
    X_train = np.vstack([train_class1, train_class2])
    y_train = np.hstack([np.zeros(len(train_class1)), np.ones(len(train_class2))])
    
    # Load test data
    test_class1 = load_images_from_folder(os.path.join(data_dir, 'test', class1_name))
    test_class2 = load_images_from_folder(os.path.join(data_dir, 'test', class2_name))
    
    X_test = np.vstack([test_class1, test_class2])
    y_test = np.hstack([np.zeros(len(test_class1)), np.ones(len(test_class2))])
    
    return X_train, y_train, X_test, y_test, class1_name, class2_name

def plot_images(images, titles, rows=1, cols=5, figsize=(15, 3)):
    """Plot multiple images"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        if idx >= len(axes):
            break
        
        # Reshape flat image back to 32x32x3
        img_reshaped = img.reshape(32, 32, 3)
        # Clip values to [0, 1] range
        img_reshaped = np.clip(img_reshaped, 0, 1)
        
        axes[idx].imshow(img_reshaped)
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('support_vectors_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Configuration - Change these based on your entry number
    # Example: If last 2 digits are 42, class1=2, class2=3
    ENTRY_LAST_2_DIGITS = 3  # CHANGE THIS
    class1_idx = ENTRY_LAST_2_DIGITS % 10
    class2_idx = (ENTRY_LAST_2_DIGITS + 1) % 10
    
    DATA_DIR = 'data'
    
    print("=" * 60)
    print("SVM Binary Classification")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test, class1_name, class2_name = load_binary_data(
        DATA_DIR, class1_idx, class2_idx
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # ========== Part 1: Linear Kernel SVM ==========
    print("\n" + "=" * 60)
    print("Part 1: Linear Kernel SVM (C=1.0)")
    print("=" * 60)
    
    svm_linear = SupportVectorMachine()
    
    print("\nTraining linear SVM...")
    start_time = time.time()
    svm_linear.fit(X_train, y_train, kernel='linear', C=1.0)
    train_time_linear = time.time() - start_time
    print(f"Training time: {train_time_linear:.2f} seconds")
    
    # (a) Number of support vectors
    n_sv_linear = len(svm_linear.support_vectors)
    sv_percentage = (n_sv_linear / len(X_train)) * 100
    print(f"\nNumber of support vectors: {n_sv_linear}")
    print(f"Percentage of training samples: {sv_percentage:.2f}%")
    
    # (b) Test accuracy
    y_pred_train = svm_linear.predict(X_train)
    y_pred_test = svm_linear.predict(X_test)
    
    train_accuracy_linear = np.mean(y_pred_train == y_train) * 100
    test_accuracy_linear = np.mean(y_pred_test == y_test) * 100
    
    print(f"\nTrain accuracy: {train_accuracy_linear:.2f}%")
    print(f"Test accuracy: {test_accuracy_linear:.2f}%")
    
    # (c) Plot top-5 support vectors and weight vector
    print("\nPlotting top-5 support vectors and weight vector...")
    top5_indices = np.argsort(svm_linear.alphas)[-5:][::-1]
    top5_sv = svm_linear.support_vectors[top5_indices]
    top5_alphas = svm_linear.alphas[top5_indices]
    
    images_to_plot = list(top5_sv) + [svm_linear.w]
    titles = [f"SV {i+1} (α={alpha:.3f})" for i, alpha in enumerate(top5_alphas)]
    titles.append("Weight Vector w")
    
    plot_images(images_to_plot, titles, rows=1, cols=6, figsize=(18, 3))
    
    # ========== Part 2: Gaussian Kernel SVM ==========
    print("\n" + "=" * 60)
    print("Part 2: Gaussian Kernel SVM (C=1.0, γ=0.001)")
    print("=" * 60)
    
    svm_gaussian = SupportVectorMachine()
    
    print("\nTraining Gaussian SVM...")
    start_time = time.time()
    svm_gaussian.fit(X_train, y_train, kernel='gaussian', C=1.0, gamma=0.001)
    train_time_gaussian = time.time() - start_time
    print(f"Training time: {train_time_gaussian:.2f} seconds")
    
    # (a) Number of support vectors
    n_sv_gaussian = len(svm_gaussian.support_vectors)
    print(f"\nNumber of support vectors: {n_sv_gaussian}")
    print(f"Comparison with linear: {n_sv_gaussian} vs {n_sv_linear}")
    
    # Find matching support vectors
    matching_sv = 0
    for sv_gauss in svm_gaussian.support_vectors:
        for sv_lin in svm_linear.support_vectors:
            if np.allclose(sv_gauss, sv_lin):
                matching_sv += 1
                break
    
    print(f"Matching support vectors: {matching_sv}")
    
    # (b) Test accuracy
    y_pred_test_gaussian = svm_gaussian.predict(X_test)
    test_accuracy_gaussian = np.mean(y_pred_test_gaussian == y_test) * 100
    
    print(f"\nTest accuracy: {test_accuracy_gaussian:.2f}%")
    
    # (c) Plot top-5 support vectors
    print("\nPlotting top-5 support vectors for Gaussian kernel...")
    top5_indices_gauss = np.argsort(svm_gaussian.alphas)[-5:][::-1]
    top5_sv_gauss = svm_gaussian.support_vectors[top5_indices_gauss]
    top5_alphas_gauss = svm_gaussian.alphas[top5_indices_gauss]
    
    titles_gauss = [f"SV {i+1} (α={alpha:.3f})" for i, alpha in enumerate(top5_alphas_gauss)]
    plot_images(top5_sv_gauss, titles_gauss, rows=1, cols=5, figsize=(15, 3))
    
    # (d) Comparison
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"\nLinear Kernel:")
    print(f"  - Support Vectors: {n_sv_linear} ({sv_percentage:.2f}%)")
    print(f"  - Test Accuracy: {test_accuracy_linear:.2f}%")
    print(f"  - Training Time: {train_time_linear:.2f}s")
    
    print(f"\nGaussian Kernel:")
    print(f"  - Support Vectors: {n_sv_gaussian}")
    print(f"  - Test Accuracy: {test_accuracy_gaussian:.2f}%")
    print(f"  - Training Time: {train_time_gaussian:.2f}s")
    
    accuracy_diff = test_accuracy_gaussian - test_accuracy_linear
    print(f"\nAccuracy difference: {accuracy_diff:+.2f}%")

if __name__ == "__main__":
    main()