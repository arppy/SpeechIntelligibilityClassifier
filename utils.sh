rename 's/_DeepFilterNet3//' *.wav

patterns=("C_001" "C_003" "C_006" "C_010" "C_016" "C_018" "C_026" "C_027" "C_029" "C_031" "C_033" "C_036" "C_040" "C_041" "C_042" "C_043" "C_048")

patterns=("C_001" "C_003" "C_006" "C_010" "C_016" "C_018" "C_026" "C_027" "C_029")
patterns=("C_031" "C_033" "C_036" "C_040" "C_041" "C_042" "C_043" "C_048")

patterns=("C_001" "C_003" "C_006" "C_010" "C_016" "C_018")
patterns=("C_026" "C_027" "C_029")
patterns=("C_031" "C_033" "C_036")
patterns=("C_040" "C_041" "C_042" "C_043" "C_048")

patterns=("CF014 CM013 M001")
folder="deep-filter"; { for pattern in "${patterns[@]}"; do printf '%s\n' "$folder"/${pattern}*; done; } | zip -@j $folder".zip"


for f in Torgo/Torgo-wavlm/*; do base=$(basename "$f"); echo "${base%%_*}" ; done | sort -u
#F01 F03 F04 FC01 FC02 FC03 M01 M02 M03 M04 M05 MC01 MC02 MC03 MC04