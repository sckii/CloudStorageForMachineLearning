def get_period_by_window(time_window, windowSize):
    first_period_week = time_window
    last_period_week = first_period_week + windowSize
    return [first_period_week, last_period_week]

def get_period_stamps(time_window, windowSize):
    firstPeriod, lastPeriod = get_period_by_window(time_window, windowSize)
    firstPeriodLAbel, lastPeriodLabel = get_period_by_window(lastPeriod, windowSize)
    return [firstPeriod, lastPeriod, firstPeriodLAbel, lastPeriodLabel]

def get_all_windows(time_window, windowSize, stepsToTake):
    steps = stepsToTake * time_window

    initialTrain, endTrain, initialTrainLabel, endTrainLabel = get_period_stamps(steps, windowSize)
    (initialEvaluation, endEvaluation, initialEvaluationLabel, endEvaluationLabel) = get_period_stamps(initialTrainLabel, windowSize)
    
    endEvaluationLabel = initialEvaluationLabel + stepsToTake
    
    return [
        initialTrain,
        endTrain,
        initialTrainLabel,
        endTrainLabel,
        initialEvaluation,
        endEvaluation,
        initialEvaluationLabel,
        endEvaluationLabel,
    ]
