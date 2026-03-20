import 'package:flutter/material.dart';

import '../config/theme.dart';
import '../models/report.dart';
import 'traffic_light.dart';

/// Card that shows the screening result for one eye, including
/// the DR grade, confidence percentage, and traffic-light indicator.
class ResultCard extends StatelessWidget {
  final EyeResult result;

  const ResultCard({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final riskColor = _colorForRisk(result.riskLevel);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            // Traffic light
            TrafficLightIndicator(
              riskLevel: result.riskLevel,
              size: 90,
            ),
            const SizedBox(width: 20),

            // Details
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Eye label
                  Text(
                    result.eye == 'left' ? 'Left Eye' : 'Right Eye',
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                      color: NetraTheme.textSecondary,
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: 6),

                  // DR label
                  Text(
                    result.drLabel,
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                      color: riskColor,
                    ),
                  ),
                  const SizedBox(height: 4),

                  // Grade
                  Text(
                    'Grade ${result.drGrade} / 4',
                    style: const TextStyle(
                      fontSize: 14,
                      color: NetraTheme.textSecondary,
                    ),
                  ),
                  const SizedBox(height: 8),

                  // Confidence bar
                  _ConfidenceBar(
                    confidence: result.confidence,
                    color: riskColor,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _colorForRisk(RiskLevel level) {
    switch (level) {
      case RiskLevel.low:
        return NetraTheme.riskLow;
      case RiskLevel.moderate:
        return NetraTheme.riskModerate;
      case RiskLevel.high:
        return NetraTheme.riskHigh;
    }
  }
}

class _ConfidenceBar extends StatelessWidget {
  final double confidence;
  final Color color;

  const _ConfidenceBar({required this.confidence, required this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Confidence',
              style: TextStyle(fontSize: 12, color: NetraTheme.textSecondary),
            ),
            Text(
              '${(confidence * 100).toStringAsFixed(1)}%',
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: confidence,
            backgroundColor: color.withOpacity(0.15),
            valueColor: AlwaysStoppedAnimation<Color>(color),
            minHeight: 6,
          ),
        ),
      ],
    );
  }
}
