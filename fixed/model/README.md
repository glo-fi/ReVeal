I *think* we just need the GGNN and representation learning model for now.

Should try and effectively 'unit test' each model while keeping track of requirements.

On closer inspection, the "representation learning model" is just a wrapper around the "metric learning model", with almost all of its functionality repeated and tremendous amounts of duplicate code. Although the wrapper provides *some* modifications to the code, these are extremely small and should have been handled in an entirely different manner. The actual training regime/loss function does not differ between the two, so let's just use the metric learning model.

I think it's becoming obvious that this code has just been copied and pasted from various sources with very few original contributions.