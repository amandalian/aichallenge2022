# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Performs multiple object tracking for detected bboxes."""

from typing import Any, Dict

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.dabble.trackingv1.detection_tracker import (
    DetectionTracker,
)


class Node(AbstractNode):
    """Uses bounding boxes detected by an object detector model to track
    multiple objects. :mod:`dabble.tracking` is a useful alternative to
    :mod:`model.fairmot` and :mod:`model.jde` as it can track bounding boxes
    detected by the upstream object detector and is not limited to only
    ``"person"`` detections.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.tracker = DetectionTracker(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Tracks detection bounding boxes.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img", "bboxes", and
                "bbox_scores.

        Returns:
            outputs (Dict[str, Any]): Tracking IDs of bounding boxes.
            "obj_attrs" key is used for compatibility with draw nodes.
        """
        # Potentially use frame_rate here too since IOUTracker has a
        # max_time_lost
        metadata = inputs.get("mot_metadata", {"reset_model": False})
        reset_model = metadata["reset_model"]
        if reset_model:
            self._reset_model()

        track_ids = self.tracker.track_detections(inputs)

        return {"obj_attrs": {"ids": track_ids}}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"tracking_type": str, "iou_threshold": float, "max_lost": int}

    def _reset_model(self) -> None:
        """Creates a new instance of DetectionTracker."""
        self.logger.info(f"Creating new {self.config['tracking_type']} tracker...")
        self.tracker = DetectionTracker(self.config)
