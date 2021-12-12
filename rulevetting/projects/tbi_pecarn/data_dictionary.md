# Feature Guide for Processed Dataset

*Note that not all of these columns will appear in the final dataset, depending on which judgement calls have been activated.*

\*\* indicates a feature whose coding/values or presence depends on a judgement call

\*\*\* indicates an **umbrella** feature, meaning that it's a feature that has sub-features whose
	values depend on the umbrella feature.
	For example: the location of basilar skull fracture occurrence would be a sub feature for a larger indicator on 
	whether there was basilar skull fracture in the first place;
	as a default, if there is no basilar skull fracture, the sub-feature location is coded as missing. 
	For these features, our strategies for imputing missing values are given by	three possible decisions,
	each one keeping a different subset of the original columns:

1. First,  could make a new binary umbrella variable where each observation is 1 if any of the umbrella or 
its subfeatures are positively marked (0 otherwise), and keep only this new column, a process we call
"**unioning**".

2. Second, some subfeatures may be more important than other subfeatures, so we can union
with respect to these more important subfeatures but keep these subfeatures as separate columns, and drop
the subfeatures deemed less important.

3. Third, we can just keep all of the columns but drop any observation with a missing value among any of the subfeatures.


These three decisions give different *strategies* for the umbrella variables, but regardless of the strategy, any remaining
missing values are always dropped. For each umbrella feature, we mark which strategy is used by default.
	
Note: for any feature that is not already binary, we provide the categories as originally encoded, 
but note that these categorical variables are one-hot encoded in the final dataset.



* InjuryMech**
	- How the injury occurred
	- Coded as: (1 Occupant in motor vehicle collision (MVC);
			2 Pedestrian struck by moving vehicle;
			3 Bike rider struck by automobile;
			4 Bike collision or fall from bike while riding;
			5 Other wheeled transport crash
			6 Fall to ground from standing/walking/running
			7 Walked or ran into stationary object
			8 Fall from an elevation
			9 Fall down stairs
			10 Sports
			11 Assault
			12 Object struck head - accidentally)
	- Inclusion by Judgement Call: step1_injMech, DEFAULT: FALSE

* High_impact_InjSev_1, High_impact_InjSev_2, High_impact_InjSev_3
	- Indicators for severity of injury mechanism (whether rated 1, 2, or 3)
	- Coded as binary for each observation

* Amnesia_verb_0, Amnesia_verb_1, Amnesia_verb_91
	- Indicators for if you don't have Amnesia (Amnesia_verb_0), you do have Amnesia (Amnesia_verb_1),
		or you are marked as pre-verbal/non-verbal (Amnesia_verb_91)

* LOCSeparate**, ***
	- History of loss of consciousness?
	- Coded as: (1 yes or suspected; 0 otherwise)
		- Alternative coding by judgement call: 0 otherwise, 1 yes, 2 suspected
	- Default strategy: Strategy 3

	* LOCLen

		- Duration of loss of consciousness
		- Coded as: (1: <5 sec, 2: 5 sec - 1 min, 3: 1 - 5 min, 4: > 5 min, or missing)

* Seiz ***
	- Whether there was a seizure
	- Default strategy: Strategy 2 (SeizOccur is dropped)

	* SeizLen
		- Duration of the seizure
		- Coded as: (1: <1 min, 2: 1 sec - < 5 min, 3: 5 - 15 min, 4: > 15 min, or missing)

	* SeizOccur
		- How long after injury did the seizure occur
		- Coded as: (1: Immediately on contact, 2: Within 30 minutes of injury, 3: >30 minutes after injury, or missing)

* ActNorm
	- Whether the parent thinks the child is acting normally

* HA_verb ***
	- Whether there was a headache at the time of evaluation, or
	the child is preverbal/nonverbal
	- Coded as: (0: no, 1: yes, 91: preverbal/nonverbal)
	- Default strategy: Strategy 2 (HAStart is dropped)

	* HASeverity
		- Ranking of severity of headache
		- Coded as: (1: Mild, 2: Moderate, 3: Severe, or missing)

	* HAStart
		- How long after injury did headache start
		- Coded as: (1: Before injury, 2: within 1 hr of event, 3: 1-4 hours after, 4: >4 hours after, or missing) 


* Vomit ***
	- Whether the individual vomited after the injury
	- Default strategy: Strategy 1

	* VomitStart
		-When did the vomiting start
		-Coded as: (1: Before injury, 2: Within 1 hour after, 3: 1-4 hours after, 4: >4 hours after, or missing)

	* VomitLast
		-How long before eval was last vomit
		-Coded as: (1: <1 hour before eval, 2: 1-4 hours before eval, 3: > 4 hours before eval, or missing)

	* VomitNbr
		-How many vomiting episodes were there
		-Coded as: (1: Once, 2: Twice, 3: >2 times, or missing)

* GCSEye
	- GCS eye score, either 3 or 4

* GCSVerbal
	- GCS verbal score, either 4 or 5

* GCSMotor 
	- GCS motor score, either 5 or 6

* GCSTotal**
	- GCS total score, either 15 or 14
	- Not included by default, inclusion by judgement call 'GCS'

* AMS***
	- GCS < 15, or other signs of altered mental status
	- Default strategy: Strategy 3

	* AMSAgitated
		- Whether they seem agitated as a reason for AMS
		- Coded as: (0: no, 1: yes, or missing)

	* AMSSleep
		- Whether they seem sleepy as a reason for AMS
		- Coded as: (0: no, 1: yes, or missing)

	* AMSSlow
		- Whether they seem slow to respond as a reason for AMS
		- Coded as: (0: no, 1: yes, or missing)

	* AMSRepeat
		- Whether they ask repetitive questions as a reason for AMS
		- Coded as: (0: no, 1: yes, or missing)

	* AMSOth
		- Whether there is any other reason to suspect AMS
		- Coded as: (0: no, 1: yes, or missing)

* SFxPalp**, ***
	- Palpable skull fracture?
	- Coded as (1: yes or unclear, 0: otherwise)
		- Alternative coding (1: yes, 0: otherwise, 2: unclear) by different judgement call
	- Default strategy: Strategy 3

	* SFxPalpDepress
		- Whether the palpable skull fracture feels depressed
		- Coded as: (0: no, 1: yes, or missing)

* FontBulg
	- Anterior fontanelle bulging? Either yes or no/closed (meaning you're older)

* SFxBas***
	- Signs of basilar skull fracture?
	- Default strategy: Strategy 3

	* SFxBasHem
		- hemotympanum?
		- Coded as: (0: no, 1: yes, or missing)

	* SFxBasOto
		- CSF otorrhea?
		- Coded as: (0: no, 1: yes, or missing)

	* SFxBasPer
		- periorbital ecchymosis (raccoon eyes)?
		- Coded as: (0: no, 1: yes, or missing)

	* SFxBasRet
		- retroauricular ecchymosis (battle's sign)?
		- Coded as: (0: no, 1: yes, or missing)

	* SFxBasRhi
		- CSF rhinorrhea?
		- Coded as: (0: no, 1: yes, or missing)

* Hema***
	- Raised scalp hematoma or swelling?
	- Default strategy: Strategy 3

	* HemaLoc
		- Location?
		- Coded as: (1: Frontal, 2: Occipital, 3: Parietal/Temporal, or missing)

	* HemaSize
		- Size?
		- Coded as: (1: small < 1cm, medium 1 - 3cm, large, >3cm, or missing)

* Clav***
	- Evidence of trauma above the clavicles?
	- Default strategy: Strategy 3

	* ClavFace
		- Was it on the face?
		- Coded as: (0: no, 1: yes, or missing)

	* ClavNeck
		- Was it on the neck?
		- Coded as: (0: no, 1: yes, or missing)

	* ClavFro
		- Was it on the scalp, but frontal?
		- Coded as: (0: no, 1: yes, or missing)

	* ClavOcc
		- Was it on the scalp, but occipital?
		- Coded as: (0: no, 1: yes, or missing)

	* ClavPar
		- Was it on the scalp, but parietal?
		- Coded as: (0: no, 1: yes, or missing)

	* ClavTem
		- Was it on the scalp, but temporal?
		- Coded as: (0: no, 1: yes, or missing)

* NeuroD***
	- Evidence of neurological deficit, besides altered mental status?
	- Default strategy: Strategy 3

	* NeuroDMotor
		- Motor deficit?
		- Coded as: (0: no, 1: yes, or missing)

	* NeuroDSensory
		- Sensory deficit?
		- Coded as: (0: no, 1: yes, or missing)

	* NeuroDCranial
		- Cranial nerve?
		- Coded as: (0: no, 1: yes, or missing)

	* NeuroDReflex
		- Reflex issue?
		- Coded as: (0: no, 1: yes, or missing)

	* NeuroDOth
		- other neurological deficit?
		- Coded as: (0: no, 1: yes, or missing)

* OSI***
	- Evidence of substantial non-head injuries
	- Default strategy: Strategy 3

	* OSIExtremity
		- an injury on the extremities?
		- Coded as: (0: no, 1: yes, or missing)

	* OSICut
		- laceration requiring OR repair
		- Coded as: (0: no, 1: yes, or missing)

	* OSICspine
		- injury to c-spine
		- Coded as: (0: no, 1: yes, or missing)

	* OSIFlank
		- injury to chest/back/flank
		- Coded as: (0: no, 1: yes, or missing)

	* OSIAbdomen
		- intra-abdominal injury
		- Coded as: (0: no, 1: yes, or missing)

	* OSIPelvis
		- pelvis injury
		- Coded as: (0: no, 1: yes, or missing)

	* OSIOth
		- another substantial non-head injury
		- Coded as: (0: no, 1: yes, or missing)

* Drugs
	- Clinical suspicion for alcohol or drug intoxication (not by laboratory testing)?

* AgeinYears
	- the age of the individual in years

* Gender
	- encoded as either male or female

* Race
	- either encoded as White, Black, Asian, American Indian/Alaska Native, Pacific Islander

* Outcome

	- a pooled outcome variable that is completely binary (1 or 0)
	- 1 if the observation had neurological surgery, intubated > 24 hours,
	death due to TBI or in the ED, hospitalized for >= 2 nights due to injury, 
	or already marked as 1 on PosIntFinal (Kupperman et al.'s definition of ciTBI)





	







	
		
	


	


