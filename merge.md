# 1. Open terminal in project root:

# 2. Run this exact sequence to merge testing into main
git fetch origin
git checkout main
git pull origin main
git merge testing
git push origin main

# 3. If git says there are conflicts:
Fix the conflicted files in the editor.

Then run:

git add .
git commit -m "Resolve merge conflicts: testing into main"
git push origin main