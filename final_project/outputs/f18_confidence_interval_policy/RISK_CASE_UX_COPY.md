# 위험 케이스 UX / 문구 설계

## 목적
- 최종 가격 모델 `canonical_F18_reference_huber_010`의 예측값은 그대로 둡니다.
- `F41_confidence_interval_policy`는 위험 케이스를 표시하고 예상 오차 범위를 안내하는 UX 장치로만 사용합니다.
- 지도 탐색을 방해하지 않기 위해 기본 marker에는 경고를 붙이지 않고, detail drawer에서만 노출합니다.

## 적용 위치
| UI 단위 | 적용 | 원칙 |
| --- | --- | --- |
| complex marker | 기본 미노출 | marker는 가격/세대수 scan 우선 |
| selected marker | 선택적 약한 border | high risk일 때만 지도 맥락 보조 |
| detail drawer | 주 노출 | 예측가 바로 아래 compact row |
| mobile sheet | 주 노출 | 첫 화면에는 한 줄 요약, 상세 문구는 접기 |

## 위험 상태
| 상태 | 조건 | chip | 기본 문구 | 보조 문구 | 구간 |
| --- | --- | --- | --- | --- | --- |
| `normal` | `confidence_tier in [high, medium] and resid_risk_tier not in [high, unknown]` | 예측 신뢰도 보통 | 최근 비교 이력을 기준으로 예측 범위를 산정했습니다. | 실제 거래 조건에 따라 차이가 날 수 있습니다. | show p90 |
| `low_history` | `confidence_tier == low and resid_risk_tier not in [high, unknown]` | 비교 이력 부족 | 이 단지는 비교 이력이 적어 예측 범위를 넓게 봐야 합니다. | 최근 실거래와 층, 동, 수리 상태를 함께 확인하세요. | show p90 and make source visible |
| `high_risk` | `resid_risk_tier == high` | 큰 오차 주의 | 과거 오차 패턴상 실제 가격과 크게 차이날 가능성이 평균보다 높습니다. | 가격은 참고값으로 보고 최근 실거래를 우선 확인하세요. | show p95 by default, with p90 available as normal range |
| `unknown` | `resid_risk_tier == unknown or no matching confidence report row before global fallback` | 신뢰도 확인 필요 | 이 단지의 오차 이력이 부족해 보수적으로 해석해야 합니다. | 지역 기준 범위로 대체 표시합니다. | show fallback p95 |

## Detail Drawer 구성
```text
예측가 8.4억
[큰 오차 주의] 예상 범위 7.1억 ~ 9.8억
과거 오차 패턴상 실제 가격과 크게 차이날 가능성이 평균보다 높습니다.
기준: 단지 기준 p95 · 최근 실거래와 층/동/수리 상태 확인 필요
```

일반 상태:
```text
예측가 8.4억
[예측 신뢰도 보통] 예상 범위 7.3억 ~ 9.5억
최근 비교 이력을 기준으로 예측 범위를 산정했습니다.
기준: 단지 기준 p90
```

## 문구 카탈로그
| key | text |
| --- | --- |
| section_title | 예측 신뢰도 |
| range_label | 예상 범위 |
| source_label | 기준 |
| range_helper | 범위는 과거 예측 오차의 p90/p95를 현재 예측가에 적용한 값입니다. |
| disclaimer | 실거래 신고 지연, 층/동/향, 수리 상태, 특수거래는 별도 확인이 필요합니다. |

## 구간 계산
```text
interval_pct = p90_abs_pct_error 또는 p95_abs_pct_error
lower = predicted_price * max(0, 1 - interval_pct)
upper = predicted_price * (1 + interval_pct)
```

기본은 p90입니다. `high_risk`, `unknown`, 또는 보수적 안내가 필요한 화면에서는 p95를 우선 표시합니다.

## Fallback 기준
| 우선순위 | 기준 | 표시 문구 |
| ---: | --- | --- |
| 1 | complex | 단지 기준 |
| 2 | legal_dong | 동 기준 |
| 3 | sgg | 구 기준 |
| 4 | sido | 시도 기준 |
| 5 | global | 전체 기준 |

## 검증 수치
| metric | value |
| --- | ---: |
| recent_holdout rows | 209468 |
| high-risk rows | 28738 |
| high-risk coverage | 13.7195% |
| high-risk >20% error rate | 8.2991% |
| overall >20% error rate | 5.1225% |
| lift vs overall | 1.62x |

## Global fallback 값
| metric | value |
| --- | ---: |
| source_until_ym | 2025-12 |
| rows | 1893038 |
| p90 | 16.3597% |
| p95 | 22.1796% |

## Tier 중앙값
| tier | groups | median p90 | median p95 |
| --- | ---: | ---: | ---: |
| high | 2062 | 14.2278% | 18.6173% |
| low | 29328 | 20.8403% | 23.9938% |
| medium | 11723 | 14.2945% | 18.3038% |

## UX Guardrails
- 경고 문구는 가격을 덮지 않고, 가격 바로 아래 한 줄 요약으로 시작합니다.
- 색만으로 상태를 전달하지 않습니다. chip text, border, icon 또는 위치 정보를 함께 씁니다.
- 큰 경고 card를 추가하지 않습니다. detail drawer 안의 compact row로 유지합니다.
- `ranking`, `favorite`, `alarm`, `recommendation` 흐름으로 확장하지 않습니다.
- public API URL/field/unit 변경은 없습니다.

## 구현 handoff
- UI component candidate: `PredictionConfidenceRow` inside complex detail drawer.
- Input candidate: predicted price, selected confidence report row, `resid_risk_tier`, display source level.
- Test seam candidate: risk state mapper and interval formatter.
- First RED candidate: high risk + p95 case renders `큰 오차 주의`, p95 interval, and source label without changing predicted price.
