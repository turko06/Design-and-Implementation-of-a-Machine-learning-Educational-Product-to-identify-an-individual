import os
import segno

# Get the repository name from the environment variable
repo_name = "Design-and-Implementation-of-a-Machine-learning-Educational-Product-to-identify-an-individual"

# Define the web address
web_address = f"https://turko06.github.io/{repo_name}"

# Generate the QR code
qr = segno.make(web_address)

# Save the QR code as an image
qr.save("src/mdbook-qr-code.png",scale=10)

print("QR code generated and saved successfully.")