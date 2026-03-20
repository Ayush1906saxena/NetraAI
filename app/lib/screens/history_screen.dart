import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../config/theme.dart';
import '../models/screening.dart';
import '../services/api_service.dart';

class HistoryScreen extends ConsumerStatefulWidget {
  const HistoryScreen({super.key});

  @override
  ConsumerState<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends ConsumerState<HistoryScreen> {
  final _searchCtrl = TextEditingController();
  final _api = ApiService();

  List<Screening> _screenings = [];
  List<Screening> _filtered = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadScreenings();
    _searchCtrl.addListener(_applyFilter);
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadScreenings() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final list = await _api.getRecentScreenings(limit: 100);
      setState(() {
        _screenings = list;
        _filtered = list;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _loading = false;
        _error = 'Could not load screenings.';
      });
    }
  }

  void _applyFilter() {
    final query = _searchCtrl.text.toLowerCase().trim();
    if (query.isEmpty) {
      setState(() => _filtered = _screenings);
      return;
    }
    setState(() {
      _filtered = _screenings.where((s) {
        return s.patient.name.toLowerCase().contains(query) ||
            s.patient.phone.contains(query) ||
            s.id.contains(query);
      }).toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Screening History')),
      body: Column(
        children: [
          // Search bar
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
            child: TextField(
              controller: _searchCtrl,
              decoration: InputDecoration(
                hintText: 'Search by name, phone, or ID...',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: _searchCtrl.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          _searchCtrl.clear();
                        },
                      )
                    : null,
              ),
            ),
          ),

          // Content
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : _error != null
                    ? _ErrorState(
                        message: _error!,
                        onRetry: _loadScreenings,
                      )
                    : _filtered.isEmpty
                        ? const _EmptySearch()
                        : RefreshIndicator(
                            onRefresh: _loadScreenings,
                            child: ListView.builder(
                              padding: const EdgeInsets.only(bottom: 24),
                              itemCount: _filtered.length,
                              itemBuilder: (context, index) {
                                final s = _filtered[index];
                                return _ScreeningTile(
                                  screening: s,
                                  onTap: () {
                                    if (s.resultId != null) {
                                      context.pushNamed('results',
                                          pathParameters: {
                                            'screeningId': s.id
                                          });
                                    }
                                  },
                                );
                              },
                            ),
                          ),
          ),
        ],
      ),
    );
  }
}

// ── Sub-widgets ────────────────────────────────────────────────────────

class _ScreeningTile extends StatelessWidget {
  final Screening screening;
  final VoidCallback onTap;

  const _ScreeningTile({required this.screening, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final s = screening;
    final statusColor = _statusColor(s.status);

    return Card(
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          child: Row(
            children: [
              // Avatar
              CircleAvatar(
                radius: 24,
                backgroundColor: NetraTheme.primaryBlue.withOpacity(0.1),
                child: Text(
                  s.patient.name.isNotEmpty
                      ? s.patient.name[0].toUpperCase()
                      : '?',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: NetraTheme.primaryBlue,
                  ),
                ),
              ),
              const SizedBox(width: 14),

              // Details
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      s.patient.name,
                      style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: NetraTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Age ${s.patient.age}  |  ${s.patient.phone}',
                      style: const TextStyle(
                        fontSize: 13,
                        color: NetraTheme.textSecondary,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _formatDate(s.createdAt),
                      style: const TextStyle(
                        fontSize: 12,
                        color: NetraTheme.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),

              // Status badge
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: statusColor.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  _statusLabel(s.status),
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: statusColor,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Color _statusColor(ScreeningStatus status) {
    switch (status) {
      case ScreeningStatus.completed:
        return NetraTheme.riskLow;
      case ScreeningStatus.analyzing:
      case ScreeningStatus.uploading:
        return NetraTheme.primaryLight;
      case ScreeningStatus.failed:
        return NetraTheme.riskHigh;
      case ScreeningStatus.queuedOffline:
        return NetraTheme.riskModerate;
      default:
        return NetraTheme.textSecondary;
    }
  }

  String _statusLabel(ScreeningStatus status) {
    switch (status) {
      case ScreeningStatus.completed:
        return 'Completed';
      case ScreeningStatus.analyzing:
        return 'Analyzing';
      case ScreeningStatus.uploading:
        return 'Uploading';
      case ScreeningStatus.failed:
        return 'Failed';
      case ScreeningStatus.queuedOffline:
        return 'Queued';
      case ScreeningStatus.capturing:
        return 'Capturing';
      case ScreeningStatus.created:
        return 'Created';
    }
  }

  String _formatDate(DateTime dt) {
    final months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ];
    return '${dt.day} ${months[dt.month - 1]} ${dt.year}, '
        '${dt.hour.toString().padLeft(2, '0')}:'
        '${dt.minute.toString().padLeft(2, '0')}';
  }
}

class _ErrorState extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;

  const _ErrorState({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.cloud_off, size: 48, color: NetraTheme.textSecondary),
          const SizedBox(height: 12),
          Text(message,
              style:
                  const TextStyle(fontSize: 15, color: NetraTheme.textSecondary)),
          const SizedBox(height: 16),
          OutlinedButton.icon(
            onPressed: onRetry,
            icon: const Icon(Icons.refresh),
            label: const Text('Retry'),
          ),
        ],
      ),
    );
  }
}

class _EmptySearch extends StatelessWidget {
  const _EmptySearch();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.search_off, size: 48, color: NetraTheme.textSecondary),
          SizedBox(height: 12),
          Text(
            'No matching screenings found.',
            style: TextStyle(fontSize: 15, color: NetraTheme.textSecondary),
          ),
        ],
      ),
    );
  }
}
