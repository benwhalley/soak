for f in *.rtf; do 
  pandoc "$f" -t plain | awk '{for(i=1;i<=NF;i++) print $i}' | head -n 2000 | paste -sd' ' - > "${f%.rtf}.txt"
done