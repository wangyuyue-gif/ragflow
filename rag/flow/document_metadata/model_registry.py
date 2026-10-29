#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# Business-confirmed model groups. Keep unconfirmed look-alike model names out
# of this registry: an incorrect equivalence is worse than a missing alias when
# the resulting metadata is used as a hard retrieval filter.
MODEL_GROUPS = (
    {
        "official_models": ("XC975EV", "XC975EU"),
        "aliases": ("XC975-EV", "XC975-EU", "975(EU)EV.AX01"),
        "equipment_type": "轮胎式装载机",
    },
    {
        "official_models": ("XC938U",),
        "aliases": (),
        "equipment_type": "轮胎式装载机",
    },
    {
        "official_models": ("XC975UEV",),
        "aliases": (),
        "equipment_type": "轮胎式装载机",
    },
)
