**Original features**

| Variable name | Variable description | Notes | Imputation |
|---------------|----------------------|-------|------------|
| Seiz | Post-traumatic seizure? |  | Imputed 'Unknown' values to 'No' |
| ActNorm | Does the parent think the child is acting normally / like themself? |  | Imputed 'Unknown' values to 'No' |
| Vomit | Vomiting (at any time after injury)? |  | Imputed 'Unknown' values to 'No' |
| Intubated | Is the physician's evaluation being made after the patient was intubated? |  | Imputed 'Unknown' values to 'No' |
| Paralyzed | Is the physician's evaluation being made after the patient was pharmacologically paralyzed? |  | Imputed 'Unknown' values to 'No' |
| Sedated | Is physician's evaluation being made after the patient was pharmacologically sedated? |  | Imputed 'Unknown' values to 'No' |
| AMS | GCS < 15 or other signs of altered mental status (agitated, sleepy, slow to respond, repetitive questions in the ED, other) | AMS was defined as a GCS between 3 and 14 or other signs of altered mental status (agitation, repetitive questions, sleepy, slow to respond, or other) | Imputed 'Unknown' values to 'No' |
| AMSAgitated | Other signs of altered mental status: agitated | Not applicable is marked if patient does not have GCS < 15 or other signs of altered mental status or AMS is missing. | Imputed 'Unknown', 'Not applicable' values to 'No' |
| AMSSleep | Other signs of altered mental status: sleepy | Not applicable is marked if patient does not have GCS < 15 or other signs of altered mental status or AMS is missing. | Imputed 'Not Applicable' values to 'No' |
| AMSSlow | Other signs of altered mental status: slow to respond | Not applicable is marked if patient does not have GCS < 15 or other signs of altered mental status or AMS is missing. | Imputed 'Not applicable' values to 'No' |
| AMSRepeat | Other signs of altered mental status: repetitive questions in ED | Not applicable is marked if patient does not have GCS < 15 or other signs of altered mental status or AMS is missing. | Imputed 'Not applicable' values to 'No' |
| AMSOth | Other signs of altered mental status: other | Not applicable is marked if patient does not have GCS < 15 or other signs of altered mental status or AMS is missing. | Imputed 'Not applicable' values to 'No' |
| FontBulg | Anterior fontanelle bulging? |  | Imputed 'Unknown' values to 'No' |
| SFxBas | Signs of basilar skull fracture? |  | Imputed 'Unknown' values to 'No' |
| SFxBasHem | Basilar skull fracture: hemotympanum | Not applicable is marked if signs of basilar skull fracture is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| SFxBasOto | Basilar skull fracture: CSF otorrhea | Not applicable is marked if signs of basilar skull fracture is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| SFxBasPer | Basilar skull fracture: periorbital ecchymosis (raccoon eyes) | Not applicable is marked if signs of basilar skull fracture is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| SFxBasRet | Basilar skull fracture: retroauricular ecchymosis (battle's sign) | Not applicable is marked if signs of basilar skull fracture is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| SFxBasRhi | Basilar skull fracture: CSF rhinorrhea | Not applicable is marked if signs of basilar skull fracture is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| Hema | Raised scalp hematoma(s) or swelling(s)? |  | Imputed 'Unknown' values to 'No' |
| Clav | Any evidence of trauma (including laceration, abrasion, hematoma) above the clavicles (includes neck/face/scalp)? |  | Imputed 'Unknown' values to 'No' |
| ClavFace | Trauma above the clavicles region: face | Not applicable is marked if any evidence of trauma above the clavicles is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| ClavNeck | Trauma above the clavicles region: neck | Not applicable is marked if any evidence of trauma above the clavicles is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| ClavFro | Trauma above the clavicles region: scalp-frontal | Not applicable is marked if any evidence of trauma above the clavicles is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| ClavOcc | Trauma above the clavicles region: scalp-occipital | Not applicable is marked if any evidence of trauma above the clavicles is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| ClavPar | Trauma above the clavicles region: scalp-parietal | Not applicable is marked if any evidence of trauma above the clavicles is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| ClavTem | Trauma above the clavicles region: scalp-temporal | Not applicable is marked if any evidence of trauma above the clavicles is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| NeuroD | Neurological deficit (other than mental status)? |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| NeuroDMotor | Neurological deficit: motor | Not applicable is marked if neurological deficit (other than mental status) is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| NeuroDSensory | Neurological deficit: sensory | Not applicable is marked if neurological deficit (other than mental status) is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| NeuroDCranial | Neurological deficit: cranial nerve (includes pupil reactivity) | Not applicable is marked if neurological deficit (other than mental status) is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| NeuroDReflex | Neurological deficit: reflexes | Not applicable is marked if neurological deficit (other than mental status) is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| NeuroDOth | Neurological deficit: other deficits (e.g. cerebellar, gait) | Not applicable is marked if neurological deficit (other than mental status) is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| OSI | Clinical evidence of other (non-head) substantial injuries: (includes but not limited to fractures, intra-abdominal injuries, intra-thoracic injuries and lacerations requiring operating room repair.) |  | Imputed 'Unknown' values to 'No' |
| OSIExtremity | Other (non-head) substantial injury: extremity | Not applicable is marked if other (non-head) substantial injuries is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| OSICut | Other (non-head) substantial injury: laceration requiring repair in the operating room | Not applicable is marked if other (non-head) substantial injuries is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| OSICspine | Other (non-head) substantial injury: c-spine | Not applicable is marked if other (non-head) substantial injuries is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| OSIFlank | Other (non-head) substantial injury: chest/back/flank | Not applicable is marked if other (non-head) substantial injuries is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| OSIAbdomen | Other (non-head) substantial injury:  intra-abdominal | Not applicable is marked if other (non-head) substantial injuries is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| OSIPelvis | Other (non-head) substantial injury: pelvis | Not applicable is marked if other (non-head) substantial injuries is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| OSIOth | Other (non-head) substantial injury: other | Not applicable is marked if other (non-head) substantial injuries is answered as no or missing. | Imputed 'Not applicable' values to 'No' |
| Drugs | Clinical suspicion for alcohol or drug intoxication (not by laboratory testing)? |  | Imputed 'Unknown' values to 'No' |
| outcome | Clinically-important traumatic brain injury? | Originally was named 'PosIntFinal' |  |

**Engineered features**

| Variable name | Variable description | Notes | Imputation |
|---------------|----------------------|-------|------------|
| AgeTwoPlus | Whether age < 2 | This is computed from the time of ED evaluation or injury date if ED evaluation is missing. |  |
| InjuryMech_Assault | Injury mechanism (InjuryMech) is Assault |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Bicyclist struck by automobile | Injury mechanism (InjuryMech) is Bike rider struck by automobile |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Bike collision/fall | Injury mechanism (InjuryMech) is Bike collision or fall from bike while riding |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Fall down stairs | Injury mechanism (InjuryMech) is Fall down stairs |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Fall from an elevation | Injury mechanism (InjuryMech) is Fall from an elevation |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Fall to ground standing/walking/running | Injury mechanism (InjuryMech) is Fall to ground from standing/walking/running |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Motor vehicle collision | Injury mechanism (InjuryMech) is Occupant in motor vehicle collision (MVC) |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Object struck head - accidental | Injury mechanism (InjuryMech) is Object struck head - accidental |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Other mechanism | Injury mechanism (InjuryMech) is Other mechanism |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Other wheeled crash | Injury mechanism (InjuryMech) is Other wheeled transport crash |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Pedestrian struck by moving vehicle | Injury mechanism (InjuryMech) is Pedestrian struck by moving vehicle |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Sports | Injury mechanism (InjuryMech) is Sports |  | Imputed 'Unknown' values to 'No' |
| InjuryMech_Walked/ran into stationary object | Injury mechanism (InjuryMech) is Walked or ran into stationary object |  | Imputed 'Unknown' values to 'No' |
| High_impact_InjSev_High | Severity of injury mechanism (High_impact_InjSev) is High | Motor vehicle collision with patient ejection, death of another passenger, or rollover; Pedestrian or bicyclist without helmet struck by a motorized vehicle; Falls of > 5 feet for patients 2 yrs and older; Falls of > 3 feet  < 2 yrs; Head struck by a high-impact object | Imputed 'Unknown', 'Not applicable' values to 'No' |
| High_impact_InjSev_Moderate | Severity of injury mechanism (High_impact_InjSev) is Moderate |  Any other mechanism | Imputed 'Unknown', 'Not applicable' values to 'No' |
| High_impact_InjSev_Low | Severity of injury mechanism (High_impact_InjSev) is Low | Fall from ground level (or fall to ground from standing, walking or running); Walked/ran into stationary object | Imputed 'Unknown', 'Not applicable' values to 'No' |
| Amnesia_verb_No | Does the patient have amnesia for the event? No | Pre-verbal is marked if the patient is too young to speak.  Non-verbal is marked if the patient is intubated or otherwise unable to give an understandable verbal response.  Pre-verbal and non-verbal were determined by the physician. | Imputed 'Unknown' values to 'No' |
| Amnesia_verb_Pre/Non-verbal | Does the patient have amnesia for the event? Non-verbal | Pre-verbal is marked if the patient is too young to speak.  Non-verbal is marked if the patient is intubated or otherwise unable to give an understandable verbal response.  Pre-verbal and non-verbal were determined by the physician. | Imputed 'Unknown' values to 'No' |
| Amnesia_verb_Yes | Does the patient have amnesia for the event? Yes | Pre-verbal is marked if the patient is too young to speak.  Non-verbal is marked if the patient is intubated or otherwise unable to give an understandable verbal response.  Pre-verbal and non-verbal were determined by the physician. | Imputed 'Unknown' values to 'No' |
| LOCSeparate_No | History of loss of consciousness? (LOCSeparate) No |  | Imputed 'Unknown' values to 'No' |
| LOCSeparate_Suspected | History of loss of consciousness? (LOCSeparate) Suspected |  | Imputed 'Unknown' values to 'No' |
| LOCSeparate_Yes | History of loss of consciousness? (LOCSeparate) Yes |  | Imputed 'Unknown' values to 'No' |
| LocLen_Not applicable | Duration of loss of consciousness (LocLen) Not applicable | Not applicable is marked if history of loss of consciousness is answered as no or missing. | Imputed 'Unknown' values to 'No' |
| LocLen_1-5 min | Duration of loss of consciousness (LocLen) 1-5 min |  | Imputed 'Unknown' values to 'No' |
| LocLen_5 sec - 1 min | Duration of loss of consciousness (LocLen) 5 sec - <1 min |  | Imputed 'Unknown' values to 'No' |
| LocLen_<5 sec | Duration of loss of consciousness (LocLen) <5 sec |  | Imputed 'Unknown' values to 'No' |
| LocLen_>5 min | Duration of loss of consciousness (LocLen) >5 min |  | Imputed 'Unknown' values to 'No' |
| SeizOccur_>30 minutes after injury | When did the post-traumatic seizure occur? (SeizOccur) >30 minutes after injury |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| SeizOccur_Immediately on contact | When did the post-traumatic seizure occur? (SeizOccur) Immediately on contact |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| SeizOccur_No | When did the post-traumatic seizure occur? (SeizOccur) Not applicable | Not applicable is marked if post-traumatic seizure is answered as no or missing. | Imputed 'Unknown', 'Not applicable' values to 'No' |
| SeizOccur_Within 30 minutes of injury | When did the post-traumatic seizure occur? (SeizOccur) Within 30 minutes of injury |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| SeizLen_1-5 min | Duration of post-traumatic seizure (SeizLen) 1-5 min |  | Imputed 'Unknown' values to 'No' |
| SeizLen_5-15 min | Duration of post-traumatic seizure (SeizLen) 1-5 min |  | Imputed 'Unknown' values to 'No' |
| SeizLen_<1 min | Duration of post-traumatic seizure (SeizLen) 1-5 min |  | Imputed 'Unknown' values to 'No' |
| SeizLen_>15 min | Duration of post-traumatic seizure (SeizLen) 1-5 min |  | Imputed 'Unknown' values to 'No' |
| SeizLen_Not applicable | Duration of post-traumatic seizure (SeizLen) Not applicable | Not applicable is marked if post-traumatic seizure is answered as no or missing. | Imputed 'Unknown' values to 'No' |
| HA_verb_No | Headache at time of ED evaluation? (HA_verb) No | Pre-verbal is marked if the patient is too young to speak.  Non-verbal is marked if the patient is intubated or otherwise unable to give an understandable verbal response.  Pre-verbal and non-verbal were determined by the physician. | Imputed 'Unknown' values to 'No' |
| HA_verb_Pre/Non-verbal | Headache at time of ED evaluation? (HA_verb) Pre/Non-verbal | Pre-verbal is marked if the patient is too young to speak.  Non-verbal is marked if the patient is intubated or otherwise unable to give an understandable verbal response.  Pre-verbal and non-verbal were determined by the physician. | Imputed 'Unknown' values to 'No' |
| HA_verb_Yes | Headache at time of ED evaluation? (HA_verb) Yes | Pre-verbal is marked if the patient is too young to speak.  Non-verbal is marked if the patient is intubated or otherwise unable to give an understandable verbal response.  Pre-verbal and non-verbal were determined by the physician. | Imputed 'Unknown' values to 'No' |
| HASeverity_Mild | Severity of headache (HASeverity) Mild, barely noticeable |  | Imputed 'Unknown' values to 'No' |
| HASeverity_Moderate | Severity of headache (HASeverity) Moderate |  | Imputed 'Unknown' values to 'No' |
| HASeverity_Not applicable | Severity of headache (HASeverity) Not applicable | Not applicable is marked if headache at time of ED evaluation is answered as no, pre-verbal/non-verbal, or missing. | Imputed 'Unknown' values to 'No' |
| HASeverity_Severe | Severity of headache (HASeverity) Severe, intense |  | Imputed 'Unknown' values to 'No' |
| HAStart_1-4 hrs after event | When did the headache start? (HAStart) 1-4 hrs after event |  | Imputed 'Unknown' values to 'No' |
| HAStart_>4 hrs after event | When did the headache start? (HAStart) >4 hrs after event |  | Imputed 'Unknown' values to 'No' |
| HAStart_Before head injury | When did the headache start? (HAStart) Before head injury |  | Imputed 'Unknown' values to 'No' |
| HAStart_Not applicable | When did the headache start? (HAStart) Not applicable | Not applicable is marked if headache at time of ED evaluation is answered as no, pre-verbal/non-verbal, or missing. | Imputed 'Unknown' values to 'No' |
| HAStart_Within 1 hr of event | When did the headache start? (HAStart) Within 1 hr of event |  | Imputed 'Unknown' values to 'No' |
| VomitNbr_>2 times | How many vomiting episodes? (VomitNbr) >2 times |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitNbr_No | How many vomiting episodes? (VomitNbr) | Not applicable is marked if vomiting (at any time after injury) is answered as no or missing. | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitNbr_Once | How many vomiting episodes? (VomitNbr) Once |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitNbr_Twice | How many vomiting episodes? (VomitNbr) Twice |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitStart_1-4 hrs after event | When did the vomiting start? (VomitStart) | 1-4 hrs after event | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitStart_>4 hrs after event | When did the vomiting start? (VomitStart) >4 hrs after event |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitStart_Before head injury | When did the vomiting start? (VomitStart) Before head injury |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitStart_No | When did the vomiting start? (VomitStart) | Not applicable is marked if vomiting (at any time after injury) is answered as no or missing. | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitStart_Within 1 hr of event | When did the vomiting start? (VomitStart) Within 1 hr of event |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitLast_1-4 hrs before ED | When was the last episode of vomiting? (VomitLast) 1-4 hrs before ED |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitLast_<1 hr before ED | When was the last episode of vomiting? (VomitLast) <1 hr before ED |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitLast_>4 hrs before ED | When was the last episode of vomiting? (VomitLast) >4 hrs before ED |  | Imputed 'Unknown', 'Not applicable' values to 'No' |
| VomitLast_No | When was the last episode of vomiting? (VomitLast) Not applicable | Not applicable is marked if vomiting (at any time after injury) is answered as no or missing. | Imputed 'Unknown', 'Not applicable' values to 'No' |
| SFxPalp_No | Palpable skull fracture? (SFxPalp) No |  | Imputed 'Unknown' values to 'No' |
| SFxPalp_Unclear | Palpable skull fracture? (SFxPalp) Unclear | If significant swelling or some other reason limits the physician's ability to assess for a skull fracture "Unclear exam" was marked. In the clinical prediction rule, palpable skull fracture and unclear exam were combined. | Imputed 'Unknown' values to 'No' |
| SFxPalp_Yes | Palpable skull fracture? (SFxPalp) Yes |  | Imputed 'Unknown' values to 'No' |
| SFxPalpDepress_No | If the patient has a palpable skull fracture, does the fracture feel depressed? (SFxPalpDepress) No |  | Imputed 'Unknown' values to 'No' |
| SFxPalpDepress_Not applicable | If the patient has a palpable skull fracture, does the fracture feel depressed? (SFxPalpDepress) Not applicable | Not applicable is marked if palpable skull fracture is answered as unclear, no, or missing. | Imputed 'Unknown' values to 'No' |
| SFxPalpDepress_Yes | If the patient has a palpable skull fracture, does the fracture feel depressed? (SFxPalpDepress) Yes |  | Imputed 'Unknown' values to 'No' |
| HemaLoc_Frontal | Hematoma(s) or swelling(s) location(s) involved (HemaLoc) Frontal |  | Imputed 'Unknown' values to 'No' |
| HemaLoc_Not applicable | Hematoma(s) or swelling(s) location(s) involved (HemaLoc) Not applicable | Not applicable is marked if raised scalp hematoma(s) or swelling(s) is answered as no or missing. | Imputed 'Unknown' values to 'No' |
| HemaLoc_Occipital | Hematoma(s) or swelling(s) location(s) involved (HemaLoc) Occipital |  | Imputed 'Unknown' values to 'No' |
| HemaLoc_Parietal/Temporal | Hematoma(s) or swelling(s) location(s) involved (HemaLoc) Parietal/Temporal |  | Imputed 'Unknown' values to 'No' |
| HemaSize_Large | Size (diameter) of largest hematoma or swelling (HemaSize) | Large (>3 cm) | Imputed 'Unknown' values to 'No' |
| HemaSize_Medium | Size (diameter) of largest hematoma or swelling (HemaSize) Medium (1-3 cm) |  | Imputed 'Unknown' values to 'No' |
| HemaSize_Not applicable | Size (diameter) of largest hematoma or swelling (HemaSize) Not applicable | Not applicable is marked if raised scalp hematoma(s) or swelling(s) is answered as no or missing. | Imputed 'Unknown' values to 'No' |
| HemaSize_Small | Size (diameter) of largest hematoma or swelling (HemaSize) Small (<1 cm, barely palpable) |  | Imputed 'Unknown' values to 'No' |
