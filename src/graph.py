# Plotting
import random
# Path to the base directory containing subfolders
base_folder_path = "/content/Infosys DataSet"

# List all subfolders in the base folder
folders = [folder for folder in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, folder))]

# Prepare data for plotting
folder_names = []
image_counts = []
colors = []

# Image file extensions
image_extensions = ('.jpg', '.jpeg', '.png')  # Update if needed

# Count images for each folder
for folder in folders:
    folder_path = os.path.join(base_folder_path, folder)
    num_images = sum(1 for file in os.listdir(folder_path) if file.endswith(image_extensions))

    folder_names.append(folder)
    image_counts.append(num_images)

    # Assign a random color for each folder for distinction
    colors.append((random.random(), random.random(), random.random()))  # RGB color tuple

# Plotting
fig = plt.figure(figsize=(10, 7))
plt.bar(folder_names, image_counts, color=colors)

plt.xlabel('Folder Name', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.title('Number of Images in Each Folder', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to fit the labels

plt.show()
