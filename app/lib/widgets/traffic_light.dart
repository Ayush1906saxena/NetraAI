import 'package:flutter/material.dart';

import '../config/theme.dart';
import '../models/report.dart';

/// Vertical traffic-light indicator (green / amber / red) that
/// visually communicates the DR risk level to the operator.
class TrafficLightIndicator extends StatelessWidget {
  final RiskLevel riskLevel;
  final double size;

  const TrafficLightIndicator({
    super.key,
    required this.riskLevel,
    this.size = 80,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size * 0.55,
      padding: EdgeInsets.all(size * 0.06),
      decoration: BoxDecoration(
        color: const Color(0xFF2C3E50),
        borderRadius: BorderRadius.circular(size * 0.2),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          _light(RiskLevel.high, NetraTheme.riskHigh),
          SizedBox(height: size * 0.06),
          _light(RiskLevel.moderate, NetraTheme.riskModerate),
          SizedBox(height: size * 0.06),
          _light(RiskLevel.low, NetraTheme.riskLow),
        ],
      ),
    );
  }

  Widget _light(RiskLevel level, Color activeColor) {
    final isActive = riskLevel == level;
    final diameter = size * 0.33;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 400),
      width: diameter,
      height: diameter,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: isActive ? activeColor : activeColor.withOpacity(0.15),
        boxShadow: isActive
            ? [
                BoxShadow(
                  color: activeColor.withOpacity(0.6),
                  blurRadius: 12,
                  spreadRadius: 2,
                ),
              ]
            : [],
      ),
    );
  }
}
