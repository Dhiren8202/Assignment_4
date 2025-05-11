import trimesh

# Load mesh files using correct paths
mesh_15 = trimesh.load("/home/dhiren/13_Images/Model_1.obj", force='mesh')
mesh_25 = trimesh.load("/home/dhiren/25_Images/Model_2.obj", force='mesh')

# Extract and print metrics
print("15 Views Mesh:")
print(f"  Vertices: {len(mesh_15.vertices)}")
print(f"  Faces: {len(mesh_15.faces)}")
print(f"  Bounding Box Volume: {mesh_15.bounding_box.volume:.2f}")

print("\n25 Views Mesh:")
print(f"  Vertices: {len(mesh_25.vertices)}")
print(f"  Faces: {len(mesh_25.faces)}")
print(f"  Bounding Box Volume: {mesh_25.bounding_box.volume:.2f}")

