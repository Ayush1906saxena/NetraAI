import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../config/theme.dart';
import '../models/report.dart';
import '../providers/screening_provider.dart';
import '../widgets/result_card.dart';
import '../widgets/traffic_light.dart';

class ResultsScreen extends ConsumerWidget {
  final String screeningId;

  const ResultsScreen({super.key, required this.screeningId});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(screeningProvider);
    final report = state.report;

    if (report == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Results')),
        body: const Center(child: Text('No results available.')),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Screening Results'),
        actions: [
          IconButton(
            icon: const Icon(Icons.home),
            onPressed: () => context.go('/'),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 20, 16, 32),
        children: [
          // Overall risk banner
          _OverallRiskBanner(report: report),
          const SizedBox(height: 20),

          // Per-eye results
          ...report.eyeResults.map((r) => Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: ResultCard(result: r),
              )),

          // GradCAM images
          if (report.eyeResults.any((r) => r.gradcamUrl != null)) ...[
            const SizedBox(height: 8),
            const Text(
              'AI Attention Maps (Grad-CAM)',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: NetraTheme.textPrimary,
              ),
            ),
            const SizedBox(height: 8),
            SizedBox(
              height: 180,
              child: ListView(
                scrollDirection: Axis.horizontal,
                children: report.eyeResults
                    .where((r) => r.gradcamUrl != null)
                    .map(
                      (r) => Padding(
                        padding: const EdgeInsets.only(right: 12),
                        child: Column(
                          children: [
                            ClipRRect(
                              borderRadius: BorderRadius.circular(10),
                              child: Image.network(
                                r.gradcamUrl!,
                                width: 150,
                                height: 150,
                                fit: BoxFit.cover,
                                errorBuilder: (_, __, ___) => Container(
                                  width: 150,
                                  height: 150,
                                  color: Colors.grey.shade200,
                                  child: const Icon(Icons.image_not_supported),
                                ),
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              r.eye == 'left' ? 'Left Eye' : 'Right Eye',
                              style: const TextStyle(
                                fontSize: 12,
                                color: NetraTheme.textSecondary,
                              ),
                            ),
                          ],
                        ),
                      ),
                    )
                    .toList(),
              ),
            ),
          ],

          const SizedBox(height: 16),

          // Recommendation
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        report.referralNeeded
                            ? Icons.warning_amber_rounded
                            : Icons.info_outline,
                        color: report.referralNeeded
                            ? NetraTheme.riskHigh
                            : NetraTheme.primaryBlue,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        report.referralNeeded
                            ? 'Referral Recommended'
                            : 'Recommendation',
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    report.recommendation,
                    style: const TextStyle(
                      fontSize: 14,
                      color: NetraTheme.textSecondary,
                      height: 1.5,
                    ),
                  ),
                  if (report.referralUrgency != null) ...[
                    const SizedBox(height: 8),
                    Chip(
                      label: Text(
                        'Urgency: ${report.referralUrgency}',
                        style: const TextStyle(
                          fontSize: 12,
                          color: Colors.white,
                        ),
                      ),
                      backgroundColor: NetraTheme.riskHigh,
                    ),
                  ],
                ],
              ),
            ),
          ),

          const SizedBox(height: 24),

          // Action buttons
          ElevatedButton.icon(
            onPressed: () async {
              final url = await ref
                  .read(screeningProvider.notifier)
                  .generateReport();
              if (url != null && context.mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Report generated successfully.'),
                    backgroundColor: NetraTheme.riskLow,
                  ),
                );
              }
            },
            icon: const Icon(Icons.picture_as_pdf),
            label: const Text('Generate Report'),
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: () {
              // Share functionality placeholder — in production this would
              // use the share_plus package to share the PDF or a summary.
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Sharing is not yet available in this build.'),
                ),
              );
            },
            icon: const Icon(Icons.share),
            label: const Text('Share Results'),
          ),
        ],
      ),
    );
  }
}

// ── Sub-widgets ────────────────────────────────────────────────────────

class _OverallRiskBanner extends StatelessWidget {
  final Report report;

  const _OverallRiskBanner({required this.report});

  @override
  Widget build(BuildContext context) {
    final color = _colorFor(report.overallRisk);
    final label = _labelFor(report.overallRisk);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [color.withOpacity(0.15), color.withOpacity(0.05)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          TrafficLightIndicator(riskLevel: report.overallRisk, size: 100),
          const SizedBox(width: 20),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Overall Assessment',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: NetraTheme.textSecondary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  label,
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                    color: color,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Worst DR Grade: ${report.maxDrGrade}/4',
                  style: const TextStyle(
                    fontSize: 14,
                    color: NetraTheme.textSecondary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _colorFor(RiskLevel level) {
    switch (level) {
      case RiskLevel.low:
        return NetraTheme.riskLow;
      case RiskLevel.moderate:
        return NetraTheme.riskModerate;
      case RiskLevel.high:
        return NetraTheme.riskHigh;
    }
  }

  String _labelFor(RiskLevel level) {
    switch (level) {
      case RiskLevel.low:
        return 'Low Risk';
      case RiskLevel.moderate:
        return 'Moderate Risk';
      case RiskLevel.high:
        return 'High Risk';
    }
  }
}
