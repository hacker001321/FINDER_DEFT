#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Parts of this code are adapted from TopicGPT
# https://github.com/ArikReuter/TopicGPT
# Licensed under MIT License

from .analyses_generation import generate_analyses
from .modes_generation import generate_modes
from .refinement import refine_modes
from .assignment import assign_modes
from .metrics import compute_metrics
from .utils import set_api_config

set_api_config()

__all__ = [
    "generate_analyses",
    "generate_modes",
    "refine_modes",
    "assign_modes",
    "compute_metrics",
    "set_api_config",
]
