from django.shortcuts import render
import pickle

context = """
Bhārata, also known as India, holds its historical and cultural significance deeply rooted in the country's rich heritage. The name Bhārata was formally adopted as an alternative name for India in Article 1 of the Constitution of India, which was ratified in 1950.

The origin of the name Bhārata can be traced back to the Vedic tribe called the Bharatas, who are prominently mentioned in the Rigveda as one of the original tribes of Āryāvarta. They notably played a role in the Battle of the Ten Kings. In the Mahabharata, a revered Indian epic, the Bhārata people are described as the descendants of Bharata, the son of King Dushyanta.

The earliest recorded use of Bhārata-varṣa, which translates to 'Bhārata mainland,' in a geographical context can be found in the Hathigumpha inscription of King Kharavela, dating back to the first century BCE. However, at this time, it referred to a limited area in northern India, primarily the region west of Magadha. The Mahabharata, composed between 200 BCE and 300 CE, expanded the scope of Bhārata to encompass a larger portion of North India, although much of the Deccan and South India remained excluded.

Bhārata is not just a historical term; it has been embraced as a self-ascribed name by many people across the Indian subcontinent and in the Republic of India itself. The official Sanskrit name of the country, Bhārata Gaṇarājya, incorporates this venerable name. The origins of this name can be traced to the ancient Hindu Puranas, which refer to the Indian subcontinent as Bhāratavarṣa, distinguishing it from other varṣas or continents. In the Vayu Purana, it is proclaimed that 'he who conquers the whole of Bhāratavarṣa is celebrated as a samrāta.'

Linguistically, the Sanskrit word Bhārata is a vṛddhi derivation of Bharata, originally an epithet of Agni, the god of fire. It is rooted in the Sanskrit verb 'bhr-' which means 'to bear' or 'to carry,' with a literal interpretation of 'to be maintained' in the context of fire. This term also conveys the idea of 'one who is engaged in the search for knowledge.' Interestingly, the Esperanto name for India, 'Barato,' is also derived from Bhārata.

According to the Puranas, Bhāratavarṣa is named after Bharata, the son of Rishabha, who belonged to the Solar dynasty and is described as a Kshatriya. This significant historical and cultural heritage is prominently mentioned in various Puranic texts, including the Vishnu Purana, Vayu Purana, Linga Purana, Brahmanda Purana, Agni Purana, Skanda Purana, Khanda, and Markandaya Purana, all of which use the term Bhāratavarṣa to refer to this cherished land.
"""

def home(request):
    return render(request, 'WebApp/index.html')

def getPredictions(question):

    from transformers import pipeline
    model = pipeline(model = 'deepset/roberta-base-squad2')
    
    new_y_pred = model(question=question, context=context)
    print(new_y_pred['answer'])

    return new_y_pred['answer']

def result(request):

    question = (request.POST.get('question', False))
    print(f"Fetched {question}")

    answer = getPredictions(question)
    print(answer)
    return render(request, 'WebApp/result.html', {'answer': answer})