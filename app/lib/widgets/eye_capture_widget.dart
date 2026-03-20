import 'package:flutter/material.dart';

import '../config/theme.dart';

/// Circular overlay frame that guides the operator to centre the
/// fundus image during capture.
class EyeCaptureOverlay extends StatelessWidget {
  /// Label shown above the frame, e.g. "Left Eye" or "Right Eye".
  final String label;

  /// Whether this eye has already been captured.
  final bool isCaptured;

  const EyeCaptureOverlay({
    super.key,
    required this.label,
    this.isCaptured = false,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final size = constraints.maxWidth * 0.75;

        return Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Eye label
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 22,
                fontWeight: FontWeight.w600,
                shadows: [Shadow(blurRadius: 8, color: Colors.black54)],
              ),
            ),
            const SizedBox(height: 16),

            // Circular guide frame
            Container(
              width: size,
              height: size,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color: isCaptured
                      ? NetraTheme.riskLow
                      : Colors.white.withOpacity(0.8),
                  width: 3,
                ),
              ),
              child: isCaptured
                  ? const Center(
                      child: Icon(
                        Icons.check_circle,
                        color: NetraTheme.riskLow,
                        size: 64,
                      ),
                    )
                  : Center(
                      child: Icon(
                        Icons.visibility,
                        color: Colors.white.withOpacity(0.3),
                        size: 64,
                      ),
                    ),
            ),
            const SizedBox(height: 12),

            // Helper text
            Text(
              isCaptured
                  ? 'Captured'
                  : 'Align the fundus inside the circle',
              style: TextStyle(
                color: Colors.white.withOpacity(0.8),
                fontSize: 14,
              ),
            ),
          ],
        );
      },
    );
  }
}
