echo "Test 12 seconds (full length)"
python -m cProfile -o demo_12_sec.prof demo.py --model_name biwi --wav_path "demo/wav/test.wav" --dataset BIWI --vertice_dim 70110  --feature_dim 128 --period 25 --fps 25 --train_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6" --condition M3 --subject M1

echo "Test 9 seconds"
python -m cProfile -o demo_9_sec.prof demo.py --model_name biwi --wav_path "demo/wav/test_shortened_9_second.wav" --dataset BIWI --vertice_dim 70110  --feature_dim 128 --period 25 --fps 25 --train_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6" --condition M3 --subject M1

echo "Test 7 seconds"
python -m cProfile -o demo_7_sec.prof demo.py --model_name biwi --wav_path "demo/wav/test_shortened_7_second.wav" --dataset BIWI --vertice_dim 70110  --feature_dim 128 --period 25 --fps 25 --train_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6" --condition M3 --subject M1

echo "Test 5 seconds"
python -m cProfile -o demo_5_sec.prof demo.py --model_name biwi --wav_path "demo/wav/test_shortened_5_second.wav" --dataset BIWI --vertice_dim 70110  --feature_dim 128 --period 25 --fps 25 --train_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6" --condition M3 --subject M1

echo "Test 3 seconds"
python -m cProfile -o demo_3_sec.prof demo.py --model_name biwi --wav_path "demo/wav/test_shortened_3_second.wav" --dataset BIWI --vertice_dim 70110  --feature_dim 128 --period 25 --fps 25 --train_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6" --condition M3 --subject M1

echo "Test 2 seconds"
python -m cProfile -o demo_2_sec.prof demo.py --model_name biwi --wav_path "demo/wav/test_shortened_2_second.wav" --dataset BIWI --vertice_dim 70110  --feature_dim 128 --period 25 --fps 25 --train_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6" --condition M3 --subject M1

echo "Test 1 seconds"
python -m cProfile -o demo_1_sec.prof demo.py --model_name biwi --wav_path "demo/wav/test_shortened_1_second.wav" --dataset BIWI --vertice_dim 70110  --feature_dim 128 --period 25 --fps 25 --train_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6" --condition M3 --subject M1
