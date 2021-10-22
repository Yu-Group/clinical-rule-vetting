**Original features**

| Variable name    | Variable description           | Notes                                          |
|------------------|--------------------------------|------------------------------------------------|
| AbdDistention    | Abdominal distention           |                                                |
| AbdTenderDegree  | Degree of abdominal tenderness | None/Moderates/Severe                          |
| AbdTrauma        | Abdominal wall trauma          |                                                |
| AbdomenPain      | Complaints of abdominal pain   |                                                |
| Age              |                                | In years, [0, 17]                              |
| CostalTender     | Costal margin tenderness       |                                                |
| DecreBreathSound | Decreased breath sounds        |                                                |
| DistractingPain  | Distracting painful injury     |                                                |
| GCSScore         | GCS coma scale score           | 3-15                                           |
| Hypotension      |                                | Calculated from age  + systolic blood pressure |
| LtCostalTender   | Left costal margin tenderness  |                                                |
| MOI              | Mechanism of injury            |                                                |
| RtCostalTender   | Right costal margin tenderness |                                                |
| SeatBeltSign     | Seat belt sign                 |                                                |
| Sex            |                                              |       |
| ThoracicTender | Thoracic tenderness                          |       |
| ThoracicTrauma | Evidence of Thoracic Wall Trauma             |       |
| VomitWretch    | Vomiting                                     |       |
| outcome        | Intra-abominal injury requiring intervention |       |

**Engineered features**

| Variable name    | Variable description           | Notes                                          |
|------------------|--------------------------------|------------------------------------------------|
| AbdDistention_or_AbdomenPain | Either abdominal distention or abdominal pain                          |       |
| AbdTrauma_or_SeatBeltSign_no | Either abdominal trabua or seatbelt sign |       |
| Age<2    | Whether age < 2                                     |       |
| GCSScore_Full        | Whether GCSScore = 15 |       |




