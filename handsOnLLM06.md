# 프롬프트 엔지니어링

- 프롬프트 엔지니어링: 생성된 텍스트의 품질을 향상시키기 위해 프롬프트를 설계하는 방법

## 텍스트 생성 모델 사용

### 선택

- 피운데이션 모델 선택
- 오픈 소스 모델, 클로즈드 소스(독점) 모델

### 로드

- transformers 라이브러리 사용

```jupyterpython
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
)
```

- 프롬프트 템플릿 사용

```jupyterpython
messages = [
    {"role": "user", "content": "Create a finny joke about chickens"}
]
output = pipe(messages)
print(output[0]["generated_text"])
```

```jupyterpython
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
print(prompt)
```

### 출력 제어

- temperature
    - 텍스트 생성 무작위성(창의성) 조절
    - 온도를 높일수록 더 다양한 항목 선택 - 0이면 가장 확률 높은 항목 선택
- top_p(nucleus sampling)
    - LLM이 고려할 토큰 제어하는 샘플링 기법
    - 누적 확률이 지정한 값에 도달할 때까지 토큰 선택 -> 높을수록 더 많은 토큰 선택

```jupyterpython
# 가장 확률 높은 항목 선택
print(pipe(messages, temperature=0)[0]["generated_text"])
# 더 자유롭게 선택
print(pipe(messages, temperature=0.5)[0]["generated_text"])
```

```jupyterpython
# 모든 항목 고려
print(pipe(messages, top_p=1)[0]["generated_text"])
# 누적 50%까지 고려
print(pipe(messages, top_p=0.5)[0]["generated_text"])
```

| top_p\temperature | 높음       | 낮음     |
|-------------------|----------|--------|
| 높음                | 브레인스토밍   | 번역     |
| 낮음                | 창의적인 글쓰기 | 이메일 작성 |

## 프롬프트 엔지니어링

- LLM이 원하는 응답을 하도록 프롬프트를 설계하는 기법

### 프롬프트 구성 요소

- 지시사항: 질문, 요청
- 출력 지시어: 모델의 출력 형태를 제한
- 데이터: 지시사항 관련 데이터

### 지시 기반 프롬프트

- 특정 질문 답변, 작업 해결 시 사용하는 프롬프트
- 분류, 검색, 요약, 코드 생성, 개체명 인식 등
- 구체성 - 원하는 바를 정확히 기술
- 환각 - 답변을 알 때만 생성하라고 요청
- 순서 - 시작과 끝 정보에 초점, 중간 정보는 잊힐 수 있음

## 고급 프롬프트 엔지니어링

### 잠재적 복잡성

- 페르소나: 정체성. 수행할 역할 정의
- 지시: 핵심 작업. 구체적으로, 명확하게 작성
- 문맥: 추가 정보. 작업의 맥락을 제시
- 형식: 텍스트 출력 형식. 자동화 시스템에서 사용 시 중요
- 청중: 텍스트 소비 대상. 출력 수준을 결정할 수 있음
- 어투: 텍스트 스타일.
- 데이터: 작업에 필요한 주요 데이터

### 문맥 내 학습

- 모델에게 에시 제공하는 방법
- 예시 개수에 따라 제로샷, 원샷, 퓨샷(few)

```jupyterpython
one_shot_prompt = [
    {
        "role": "user",
        "content": "A 'Solguem' is a traditional Korean string instrument. An example of a sentence that uses the word Solguem is:"
    },
    {
        "role": "assistant",
        "content": "The musician skillfully played the Solguem, filling the room with its melodious sound."
    },
    {
        "role": "user",
        "content": "To 'dopo' something is to apply something at it. An example of a sentence that uses the word dopo is:"
    }
]
print(tokenizer.apply_chat_template(one_shot_prompt, tokenize=False))
```

```jupyterpython
outputs = pipe(one_shot_prompt)
print(outputs[0]["generated_text"])
```

### 프롬프트 체인

- 프롬프트 출력을 다음 프롬프트 입력으로 사용 - 연속적인 상호작용 체인으로 문제 해결
- Ex: 제품 이름 생성 -> 제품 슬로건 생성 -> 제품 홍보문 작성

```jupyterpython
product_prompt = [
    {
        "role": "user",
        "content": "Create a chatbot name and slogan that leverages LLMs."
    }
]
outputs = pipe(product_prompt)
product_description = outputs[0]["generated_text"]
print(product_description)
```

```jupyterpython
product_prompt = [
    {
        "role": "user",
        "content": f"Generate a sales pitch for the following product: '{product_prompt}'"
    }
]
outputs = pipe(salse_prompt)
sales_pitch = outputs[0]["generated_text"]
print(sales_pitch)
```

- 어떻게 사용할 수 있을까?
    - 응답 유효성 검사: 이전 생성 출력을 다시 확인하도록 요청
    - 병렬 프롬프트: 여러 LLM에 병렬로 요청 보내고 결과를 합쳐서 출력
    - 이야기 작성: 문제를 여러 요소로 나누어 이야기를 구성

## 생성 모델을 사용한 추론

- 인간 추론을 모방해 모델 출력 향상하는 기법
    - 조금 느리지만 자기 성찰, 논리적 사고 방식을 흉내

### CoT: 답변하기 전에 생각하기

- Chain of Thought
- 바로 질문에 답을 하지 않고 그 전에 생각하게 만드는 것이 목표
- 응답 생성 전 추론 예시를 추가하거나 제로샷일 경우 단계별로 생각해보자는 내용 추가

### 자기 일관성: 출력 샘플링

- 일정 수준의 창의성을 허용한 경우, 동일한 프롬프트를 여러 번 실행하면 다른 결과 얻을 수 있음
- 생섬 모델에게 동일 프롬프트를 여러 번 요청하고 다수를 차지하는 결과를 최종 답변으로 선택

### ToT: 중간 단계 탐색

- Tree of Thought
- 여러 단계의 추론으로 나누어 문제 해결
- 중간 사고를 생성하고 평가하여 전망이 밝은 사고를 유지하고 어두운 사고를 삭제하는 방식
- 프롬프트 하나로 모방하기 위해서는 전문가 여럿이 주고받는 대화를 흉내내도록  요청하여 합의가 될 때까지 서로에게 질문하는 방식을 적용할 수 있음

## 출력 검증

- 출력을 검증하는 이유
    - 구조적 출력: 자연어가 아닌 JSON 등의 구조화된 형태
    - 유효한 출력: 둘 중 하나를 선택하라고 했는데 다른 것을 선택
    - 윤리: 욕설, 개인 정보, 편향
    - 정확성: 표준, 성능, 일관성, 환각 여부
- 출력 제어 방법
    - 예시: 기대 출력 예시를 여러 개 제공
    - 문법: 토큰 선택 과정 제어
    - 미세 튜닝: 거대 출력이 포함된 데이터에서 모델 튜닝

### 예시 제공

- 기대 출력 예시를 제공
- 프롬프트에 제시된 형식을 따를지 말지는 모델에 달려있어 지시를 따르지 않을 수 있음

### 제약 샘플링

- 퓨샷 학습은 불완전한 출력을 완전히 막을 수 없음
- 생성형 AI로 출력을 평가(검증)
- 샘플링 과정에 제약을 가하여 관심 대상만 출력하게 만들 수 있음
