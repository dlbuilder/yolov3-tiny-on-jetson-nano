#pragma once
#include "RuntimeAPI.h"

void OutToInMemChain() {}

template <typename Output, typename Input, typename ...Args>
void OutToInMemChain(Output *&output, Input *&input, Args*&... args)
{
	if (output == nullptr)
		output = new Output();
	input = (Input*)output;
	OutToInMemChain(args...);
}

template < typename Output, typename Input, typename ...Args>
void InitFilterOutputDataandChainNextInputData(Output*& output, Input* &input, Args*& ... args)
{
	OutToInMemChain(output, input, args...);
}

void InitFilterInternalData() {}
template <class Data, class ...Args>
void InitFilterInternalData(Data*&data, Args*& ... args)
{
	data = new Data();
	InitFilterInternalData(args...);
}

void InitFilter(){}
template <typename Filter, typename Internal,typename Input,typename Output>
void InitFilter(Filter*&data,Internal *&internal,Input *&input,Output *&output)
{
	data = new Filter(internal, input, output);
}

template <typename Filter, typename Internal, typename Input1, typename Input2, typename Output>
void InitFilter(Filter*&data, Internal *&internal, Input1 *&input1, Input2 *&input2, Output *&output)
{
	data = new Filter(internal, input1,input2, output);
}