#!/bin/bash

# Create a temporary file for the helm-docs output
helm-docs --output-file=README.md.suffix

# Output the <!-- BEGIN: Auto-generated helm-docs --> and <!-- END: Auto-generated helm-docs --> back into the file
# Prepend the BEGIN comment
{ echo '<!-- BEGIN: Auto-generated helm-docs -->'; cat README.md.suffix; } > temp && mv temp README.md.suffix

# Append the END comment
echo '<!-- END: Auto-generated helm-docs -->' >> README.md.suffix

# Split the original README.md at "<!-- BEGIN: Auto-generated helm-docs -->"
awk '/<!-- BEGIN: Auto-generated helm-docs -->/{exit} {print}' README.md > README.md.prefix

# Combine the files
cat README.md.prefix README.md.suffix > README.md

# Clean up temporary files
rm README.md.prefix README.md.suffix
