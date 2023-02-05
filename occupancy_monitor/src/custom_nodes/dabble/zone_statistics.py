"""
Node template for creating custom nodes.
"""

from typing import Any, Dict, List, Union

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.dabble.zoningv1.zone import Zone


class Node(AbstractNode):

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.zones = [self._create_zone(zone) for zone in self.zones]  # type: ignore

        self.occupancy = 0
        
        self.in_top = set()
        self.in_bot = set()
        self.entered_venue = 0
        self.exited_venue = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Counts all detected objects that falls within any specified zone,
        and return the total object count in each zone.
        """
        zone_counts = [0] * len(self.zones)

        cap = self.config["capacity"]
        buffer = self.config["buffer"]
        top = self.zones[0]
        bot = self.zones[1]
        btm_midpoints = inputs["btm_midpoint"]
        ids = inputs["obj_attrs"]["ids"]
        for i in range(len(btm_midpoints)):
            bbox_id = ids[i]
            # spotted in top zone
            if top.contains(btm_midpoints[i]):
                if bbox_id in self.in_top: # still walking thru top zone
                    pass
                elif bbox_id in self.in_bot: # came from bottom zone, exited venue
                    self.in_bot.remove(bbox_id)
                    self.in_top.add(bbox_id)
                    self.exited_venue += 1
                    self.occupancy -= 1
                else: # first time in frame
                    self.in_top.add(bbox_id)
            # spotted in bottom zone
            elif bot.contains(btm_midpoints[i]):
                if bbox_id in self.in_bot: # still walking thru bottom zone
                    pass
                elif bbox_id in self.in_top: # came from top zone, entered venue
                    self.in_top.remove(bbox_id)
                    self.in_bot.add(bbox_id)
                    self.entered_venue += 1
                    self.occupancy += 1
                else: # first time in frame
                    self.in_bot.add(bbox_id)

        message = "Max capacity reached." if self.occupancy >= cap else "Limited capacity available." if cap - self.occupancy <= buffer else "Safe to enter!"


        return {
            "zones": [zone.polygon_points for zone in self.zones],
            "entered": self.entered_venue,
            "exited": self.exited_venue,
            "occupancy": str(self.occupancy) + " / " + str(cap),
            "status": message
        }

    def _create_zone(self, zone: List[List[Union[float, int]]]) -> Zone:
        """Creates the appropriate Zone given either the absolute pixel values
        or % of resolution as a fraction between [0, 1].
        """
        if all(all(0 <= i <= 1 for i in coords) for coords in zone):
            # coordinates are in fraction. Use resolution to get correct coords
            zone_points = [
                self._get_pixel_coords(coords, self.resolution) for coords in zone
            ]
        elif all(
            all((isinstance(i, int) and i >= 0) for i in coords) for coords in zone
        ):
            # list is in pixel value.
            zone_points = zone  # type: ignore
        else:
            raise ValueError(
                f"Zone {zone} needs to be all pixel-wise points or all fractions "
                "of the frame between 0 and 1. Please check zone_count configs."
            )
        created_zone = Zone(zone_points)

        return created_zone

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"resolution": List[int], "zones": List[List[List[Union[int, float]]]]}

    @staticmethod
    def _get_pixel_coords(
        coords: List[Union[float, int]], resolution: List[int]
    ) -> List[int]:
        """Returns the pixel position of the zone points."""
        return [int(coords[0] * resolution[0]), int(coords[1] * resolution[1])]
