чтобы посмотреть график, я использую gnuplot и timg
./main | gnuplot -e 'set yrange [0:]; set terminal png size 1280,480; set output "out.png"; plot "-" with lines' && timg out.png
