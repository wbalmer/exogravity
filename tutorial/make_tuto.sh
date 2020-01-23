cd ./tex;
pdflatex tuto.tex

pandoc -s swap_example.tex -o swap_example.md
python ../python/md_cleaner.py swap_example.md
pandoc -s swap_example.md -o swap_example.ipynb
rm swap_example.md
mv swap_example.ipynb ../

pandoc -s on_axis_example.tex -o onaxis_example.md
python ../python/md_cleaner.py onaxis_example.md
pandoc -s onaxis_example.md -o onaxis_example.ipynb
rm onaxis_example.md
mv onaxis_example.ipynb ../

pandoc -s data_manipulation.tex -o data_manipulation.md
python ../python/md_cleaner.py data_manipulation.md
pandoc -s data_manipulation.md -o data_manipulation.ipynb
rm data_manipulation.md
mv data_manipulation.ipynb ../

pandoc -s exogravity.tex -o exogravity.md
python ../python/md_cleaner.py exogravity.md
pandoc -s exogravity.md -o exogravity.ipynb
rm exogravity.md
mv exogravity.ipynb ../

cd ../
mv ./tex/tuto.pdf ./tuto.pdf
