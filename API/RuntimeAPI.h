#pragma once

#include "../Data/BaseData.h"

template <typename RunningFilter>
void RunFilterSync(RunningFilter runningFilter)
{
	FilterStatus result = runningFilter->RunFilterCoreLogic();
}