"""
Structural Analysis Module

This module provides the core functionality for structural analysis in the
context of finite element methods or similar computational techniques.

The module includes the `IAnalysis` interface for defining different types of
analyses and the `LinearAnalysis` class for performing linear static analysis.

Modules
-------
- IAnalysis: An abstract base class (interface) that defines the required
  methods for setting up and running an analysis.
- LinearAnalysis: A concrete implementation of the `IAnalysis` interface
  for performing a linear static analysis.

Classes
--------
- IAnalysis: Interface for all analysis classes, requiring implementation
  of setup and run_analysis methods.
- LinearAnalysis: Implements the `IAnalysis` interface for linear static
  analysis using OpenSees.

This module allows for easy extension by implementing additional analysis
types that adhere to the `IAnalysis` interface.
"""

from .abstract_analysis import AbstractAnalysis
from .linear_analysis import LinearAnalysis


__all__ = ['AbstractAnalysis', 'LinearAnalysis']
