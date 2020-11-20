# SMART-Long_Sentence_TTS

SMART-Long_Sentence_TTS 모델은 10초 이하의 짧은 훈련 데이터만으로 60초 이상의 음성을 합성할 수 있는 document-level의 한국어 음성합성 모델입니다.
공개된 코드는 2020년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "소량 데이터만을 이용한 고품질 종단형 기반의 딥러닝 다화자 운율 및 감정 복제 기술 개발" 과제의 일환으로 공개된 코드입니다.

학습 단계에서는 curriculum learning을 이용하여 짧은 문장부터 학습되며 epoch이 늘어날 때마다 합습되는 음성의 길이를 늘려 긴 문장을 학습하는 것을 가능하게 합니다.

합성 단계에서는 attention masking을 통해 현재 time step에서 필요한 부분의 context만 사용하여 합성의 효율을 높이고 attention error를 줄였습니다.

추가적으로 decoder의 output을 더 robust하게 예측하기 위해 fastspeech의 duration predictor를 사용하고 본 연구에서는 현재 time step의 mel spectrogram을 생성하는데 참조해야 하는 alignment의 위치를 결정하기 위해 사용합니다.


## Requirements
- pytorch 3.6
- pytorch 1.5.0
- inflect 0.2.5
- jamo 0.4.1
- matplotlib 2.1.0
- nltk 3.5
- numba 0.48.0
- numpy 1.16.4
- Pillow 7.2.0
- librosa 0.6.0
- tensorboardX 1.8
- tensorflow 1.15.0

To install requirements:

<pre>
<code>
pip install -r requirements.txt
</code>
</pre>


## Training

To train the model(s), run this command:

<pre>
<code>
python train.py
</code>
</pre>

## Evaluation

To evaluate, run:

<pre>
<code>
python inference.py  >> log.log
</code>
</pre>


## Reference code

[1] Tacotron2 : https://github.com/NVIDIA/tacotron2

[2] FastSpeech : https://github.com/xcmyz/FastSpeech

[3] MelGAN : https://github.com/seungwonpark/melgan
