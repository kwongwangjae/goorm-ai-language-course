import matplotlib.pyplot as plt
from matplotlib import font_manager,rc
import platform
# 한글 폰트 설정
if platform.system() == 'Windows':# 윈도우인 경우
    font_path = "c:/Windows/Fonts/malgun.ttf" # 맑은 고딕 폰트 경로
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
elif platform.system() == 'Darwin': # macOS인 경우
    plt.rc('font', family='AppleGothic')
else: # Linux 등 다른 OS인 경우
    plt.rc('font', family='NanumGothic')
# 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False