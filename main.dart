/*
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(UserManagementApp());
}

class User {
  final int id;
  final String firstName;
  final String lastName;
  final String email;
  final String gender;
  final String avatar;
  final String domain;
  final bool available;

  User({
    required this.id,
    required this.firstName,
    required this.lastName,
    required this.email,
    required this.gender,
    required this.avatar,
    required this.domain,
    required this.available,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'],
      firstName: json['first_name'],
      lastName: json['last_name'],
      email: json['email'],
      gender: json['gender'],
      avatar: json['avatar'],
      domain: json['domain'],
      available: json['available'],
    );
  }
}

class UserManagementApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: UserManagementScreen(),
    );
  }
}

class UserManagementScreen extends StatefulWidget {
  @override
  _UserManagementScreenState createState() => _UserManagementScreenState();
}

class _UserManagementScreenState extends State<UserManagementScreen> {
  List<User> users = [];
  List<User> displayedUsers = [];
  List<User> teamUsers = [];
  TextEditingController searchController = TextEditingController();
  String selectedDomain = '';
  String selectedGender = '';
  bool selectedAvailability = false;

  @override
  void initState() {
    super.initState();
    loadUserList();
  }

  Future<void> loadUserList() async {
    String data = await rootBundle.loadString('assets/heliverse_mock_data.json');
    final List<dynamic> jsonData = jsonDecode(data);
    users = jsonData.map((userJson) => User.fromJson(userJson)).toList();
    filterUsers();
  }

  void filterUsers() {
    setState(() {
      displayedUsers = users.where((user) {
        final domainFilter = selectedDomain.isEmpty || user.domain == selectedDomain;
        final genderFilter = selectedGender.isEmpty || user.gender == selectedGender;
        final availabilityFilter = selectedAvailability ? user.available : true;
        final nameFilter = user.firstName.toLowerCase().contains(searchController.text.toLowerCase());
        return domainFilter && genderFilter && availabilityFilter && nameFilter;
      }).toList();
    });
  }

  void createTeam() {
    final Map<String, List<User>> domainMap = {};

    for (var user in users) {
      if (user.available) {
        if (!domainMap.containsKey(user.domain)) {
          domainMap[user.domain] = [];
        }
        domainMap[user.domain]!.add(user);
      }
    }

    final List<List<User>> availableUsers = domainMap.values.where((users) => users.length == 1).toList();
    teamUsers = availableUsers.expand((users) => users).toList();

    // Navigate to the Team Details screen after creating the team
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => TeamDetailsScreen(teamUsers),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('User Management'),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: TextField(
              controller: searchController,
              decoration: InputDecoration(labelText: 'Search by Name'),
              onChanged: (value) {
                filterUsers();
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: DropdownButtonFormField<String>(
              value: selectedDomain.isNotEmpty ? selectedDomain : null,
              hint: Text('Select Domain'),
              onChanged: (String? newValue) {
                setState(() {
                  selectedDomain = newValue!;
                  filterUsers();
                });
              },
              items: getUniqueDomains()
                  .map<DropdownMenuItem<String>>(
                    (String value) => DropdownMenuItem<String>(
                  value: value,
                  child: Text(value),
                ),
              )
                  .toList(),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: DropdownButtonFormField<String>(
              value: selectedGender.isNotEmpty ? selectedGender : null,
              hint: Text('Select Gender'),
              onChanged: (String? newValue) {
                setState(() {
                  selectedGender = newValue!;
                  filterUsers();
                });
              },
              items: ['Male', 'Female', 'Other']
                  .map<DropdownMenuItem<String>>(
                    (String value) => DropdownMenuItem<String>(
                  value: value,
                  child: Text(value),
                ),
              )
                  .toList(),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                Text('Available'),
                Checkbox(
                  value: selectedAvailability,
                  onChanged: (bool? value) {
                    setState(() {
                      selectedAvailability = value!;
                      filterUsers();
                    });
                  },
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: displayedUsers.length,
              itemBuilder: (context, index) {
                final user = displayedUsers[index];
                return ListTile(
                  title: Text('${user.firstName} ${user.lastName}'),
                  subtitle: Text('Domain: ${user.domain}, Gender: ${user.gender}, Available: ${user.available}'),
                );
              },
            ),
          ),
          ElevatedButton(
            onPressed: createTeam,
            child: Text('Add to Team'),
          ),
          if (teamUsers.isNotEmpty)
            ElevatedButton(
              onPressed: () {
                // Navigation to TeamDetailsScreen is handled in createTeam()
              },
              child: Text('View Team Details'),
            ),
        ],
      ),
    );
  }

  List<String> getUniqueDomains() {
    return users.map((user) => user.domain).toSet().toList();
  }
}

class TeamDetailsScreen extends StatelessWidget {
  final List<User> teamUsers;

  TeamDetailsScreen(this.teamUsers);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Team Details'),
      ),
      body: ListView.builder(
        itemCount: teamUsers.length,
        itemBuilder: (context, index) {
          final user = teamUsers[index];
          return ListTile(
            title: Text('${user.firstName} ${user.lastName}'),
            subtitle: Text('Domain: ${user.domain}, Gender: ${user.gender}, Available: ${user.available}'),
          );
        },
      ),
    );
  }
}
*/
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(UserManagementApp());
}

class User {
  final int id;
  final String firstName;
  final String lastName;
  final String email;
  final String gender;
  final String avatar;
  final String domain;
  final bool available;

  User({
    required this.id,
    required this.firstName,
    required this.lastName,
    required this.email,
    required this.gender,
    required this.avatar,
    required this.domain,
    required this.available,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'],
      firstName: json['first_name'],
      lastName: json['last_name'],
      email: json['email'],
      gender: json['gender'],
      avatar: json['avatar'],
      domain: json['domain'],
      available: json['available'],
    );
  }
}

class UserManagementApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: UserManagementScreen(),
    );
  }
}

class UserManagementScreen extends StatefulWidget {
  @override
  _UserManagementScreenState createState() => _UserManagementScreenState();
}

class _UserManagementScreenState extends State<UserManagementScreen> {
  List<User> users = [];
  List<User> displayedUsers = [];
  List<User> teamUsers = [];
  TextEditingController searchController = TextEditingController();
  String selectedDomain = '';
  String selectedGender = '';
  bool selectedAvailability = false;

  @override
  void initState() {
    super.initState();
    loadUserList();
  }

  Future<void> loadUserList() async {
    String data = await rootBundle.loadString('assets/heliverse_mock_data.json');
    final List<dynamic> jsonData = jsonDecode(data);
    users = jsonData.map((userJson) => User.fromJson(userJson)).toList();
    filterUsers();
  }

  void filterUsers() {
    setState(() {
      displayedUsers = users.where((user) {
        final domainFilter = selectedDomain.isEmpty || user.domain == selectedDomain;
        final genderFilter = selectedGender.isEmpty || user.gender == selectedGender;
        final availabilityFilter = selectedAvailability ? user.available : true;
        final nameFilter = user.firstName.toLowerCase().contains(searchController.text.toLowerCase());
        return domainFilter && genderFilter && availabilityFilter && nameFilter;
      }).toList();
    });
  }

  void createTeam() {
    final Map<String, List<User>> domainMap = {};

    for (var user in users) {
      if (user.available) {
        if (!domainMap.containsKey(user.domain)) {
          domainMap[user.domain] = [];
        }
        domainMap[user.domain]!.add(user);
      }
    }

    final List<List<User>> availableUsers = domainMap.values.where((users) => users.length == 1).toList();
    teamUsers = availableUsers.expand((users) => users).toList();

    // Navigate to the Team Details screen after creating the team
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => TeamDetailsScreen(teamUsers),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('User Management'),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: TextField(
              controller: searchController,
              decoration: InputDecoration(labelText: 'Search by Name'),
              onChanged: (value) {
                filterUsers();
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: DropdownButtonFormField<String>(
              value: selectedDomain.isNotEmpty ? selectedDomain : null,
              hint: Text('Select Domain'),
              onChanged: (String? newValue) {
                setState(() {
                  selectedDomain = newValue!;
                  filterUsers();
                });
              },
              items: getUniqueDomains()
                  .map<DropdownMenuItem<String>>(
                    (String value) => DropdownMenuItem<String>(
                  value: value,
                  child: Text(value),
                ),
              )
                  .toList(),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: DropdownButtonFormField<String>(
              value: selectedGender.isNotEmpty ? selectedGender : null,
              hint: Text('Select Gender'),
              onChanged: (String? newValue) {
                setState(() {
                  selectedGender = newValue!;
                  filterUsers();
                });
              },
              items: ['Male', 'Female', 'Other']
                  .map<DropdownMenuItem<String>>(
                    (String value) => DropdownMenuItem<String>(
                  value: value,
                  child: Text(value),
                ),
              )
                  .toList(),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                Text('Available'),
                Checkbox(
                  value: selectedAvailability,
                  onChanged: (bool? value) {
                    setState(() {
                      selectedAvailability = value!;
                      filterUsers();
                    });
                  },
                ),
              ],
            ),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: displayedUsers.length,
              itemBuilder: (context, index) {
                final user = displayedUsers[index];
                return ListTile(
                  title: Text('${user.firstName} ${user.lastName}'),
                  subtitle: Text('Domain: ${user.domain}, Gender: ${user.gender}, Available: ${user.available}'),
                );
              },
            ),
          ),
          ElevatedButton(
            onPressed: createTeam,
            child: Text('Add to Team'),
          ),
          if (teamUsers.isNotEmpty)
            ElevatedButton(
              onPressed: () {
                // Navigation to TeamDetailsScreen is handled in createTeam()
              },
              child: Text('View Team Details'),
            ),
        ],
      ),
    );
  }

  List<String> getUniqueDomains() {
    return users.map((user) => user.domain).toSet().toList();
  }
}

class TeamDetailsScreen extends StatelessWidget {
  final List<User> teamUsers;

  TeamDetailsScreen(this.teamUsers);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Team Details'),
      ),
      body: ListView.builder(
        itemCount: teamUsers.length,
        itemBuilder: (context, index) {
          final user = teamUsers[index];
          return ListTile(
            title: Text('${user.firstName} ${user.lastName}'),
            subtitle: Text('Domain: ${user.domain}, Gender: ${user.gender}, Available: ${user.available}'),
          );
        },
      ),
    );
  }
}
