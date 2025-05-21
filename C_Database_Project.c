#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#define TABLE_SIZE 100

typedef struct Student
{
    char name[30];
    char id[10];
    char email[30];
    char major[30];
    char phone_number[20];
    char year[5];
    struct Student *next;
} Student;
Student *HashTable[TABLE_SIZE];
int i = 0, sum = 0, currentID = 0;
void EmailGenerator(Student *NewStudent)
{
    char temp[30], out[30];
    strcpy(temp, NewStudent->name);
    int i = 0, k = 0;

    while (temp[i] != '\0')
    {
        if (i == 0)
        {
            out[k] = tolower(temp[i]);
            k++;
        }
        else if (temp[i] == ' ')
        {
            i++;
            while (temp[i] != '\0')
            {
                out[k++] = tolower(temp[i++]);
            }
            break;
        };
        i++;
    }
    int len = strlen(NewStudent->year);
    out[k++] = NewStudent->year[len - 2];
    out[k++] = NewStudent->year[len - 1];
    out[k] = '\0';
    sprintf(NewStudent->email, "%s@bard.edu", out);
}
void ID(Student *NewStudent)
{
    sprintf(NewStudent->id, "SR%03d", currentID++);
}
void init()
{
    for (; i < TABLE_SIZE; i++)
    {
        HashTable[i] = NULL;
    }
}

int hash(char *name)
{
    sum = 0;
    i = 0;
    while (name[i] != '\0')
    {
        sum += name[i];
        i++;
    }
    return sum % TABLE_SIZE;
}
void hashtable(int *index, Student *S)
{
    if (!HashTable[(*index)])
    {
        HashTable[*index] = S;
        S->next = NULL;
    }
    else
    {
        S->next = HashTable[*index];
        HashTable[*index] = S;
    }
}
void read_file(FILE *fptr)
{
    int index = 0;
    char line[200], name[30], phone_number[30], major[30], year[5];
    while (fgets(line, sizeof(line), fptr))
    {
        if (sscanf(line, "%[^,],%[^,],%[^,],%[^,\n]",
                   name, phone_number, major, year) == 4)
        {
            Student *NewStudent = (Student *)malloc(sizeof(Student));
            index = hash(name);
            hashtable(&index, NewStudent);
            strcpy(NewStudent->year, year);
            strcpy(NewStudent->name, name);
            strcpy(NewStudent->phone_number, phone_number);
            strcpy(NewStudent->major, major);
            ID(NewStudent);
            EmailGenerator(NewStudent);
        }
    }
}
Student *search(char *name_of_student)
{
    int index = 0;
    index = hash(name_of_student);
    Student *S = HashTable[index];
    char name[30];
    strcpy(name, S->name);
    while (S != NULL)
    {
        if (strlen(name) != strlen(name_of_student))
        {
            S = S->next;
        }
        else
        {
            break;
        }
    }
    if (S == NULL)
    {
        printf("\n\nSorry, %s was not found in our database\n\n", name_of_student);
        exit(1);
    }
    strcpy(name, S->name);
    char *n = name;
    char *n_o_s = name_of_student;
    int length_name = strlen(name);
    for (i = 0; i < length_name; i++, n++, n_o_s++)
    {
        if (*n == name_of_student[i])
        {
            if (i == (length_name - 1))
            {
                printf("\n\nThe student was successfully found:\n\n%s, %s, %s, %s, %s, %s ", S->name, S->id, S->email, S->phone_number, S->major, S->year);
                return S;
                break;
            }
        }
        if (*n != name_of_student[i]) // Does not work
        {
            if (S == NULL)
            {
                printf("\n\nSorry, I cannot find %s in my database\n\n", name_of_student);
                break;
            }
            S = S->next;
            strcpy(name, S->name);
            n = name;
            n_o_s = name_of_student;
            length_name = strlen(name);
            i = 0;
        }
    }
}
void add(char *name, char *phone_number, char *major, char *year)
{
    Student *NewStudent = (Student *)malloc(sizeof(Student));
    int index = 0;
    index = hash(name);
    hashtable(&index, NewStudent);
    NewStudent = HashTable[index];
    printf("\n\nYou are adding the following information to the database for %s:\n\n", name);
    strcpy(NewStudent->name, name);
    strcpy(NewStudent->phone_number, phone_number);
    strcpy(NewStudent->major, major);
    strcpy(NewStudent->year, year);
    ID(NewStudent);
    EmailGenerator(NewStudent);
    printf("%s,", NewStudent->name);
    printf("%s,", NewStudent->id);
    printf("%s,", NewStudent->email);
    printf("%s,", NewStudent->phone_number);
    printf("%s,", NewStudent->major);
    printf("%s", NewStudent->year);
}
void delete(char *name, char *phone_number, char *major, char *year)
{
    int index = 0;
    index = hash(name);
    Student *NewStudent = (Student *)calloc(1, sizeof(Student));
    NewStudent = HashTable[index];
    while (strcmp(NewStudent->name, name))
    {
        NewStudent = NewStudent->next;
        if (NewStudent == NULL)
        {
            printf("\n\nSorry, we did not find the student %s in our database\n\n", name);
            exit(1);
        }
    }
    NewStudent = NULL;
}
int main()
{
    init();
    FILE *fptr;
    fptr = fopen("Database.txt", "r");
    read_file(fptr);
    add("Artem Portsenko", "+1(413)673-1614", "Computer Engineering", "2024");
    search("Artem Protsenko");
    search("Artem Portsenko");
    fclose(fptr);
}
/*  insert_student("Artem Protsenko", "+1 (413) 673-1614", "Computer Engineering", "2024");
    insert_student("Sami Alshawi", "+1 (413) 673-1614", "Computer Engineering", "2024");
    insert_student("David Brangaitis", "+1 (413) 673-1614", "Computer Engineering", "2024");
    insert_student("Reese", "+1 (413) 673-1614", "Computer Engineering", "2024");*/
// 1. Insert all necessary information, name, phone number, major
// 2. Create the email, id
// 3. Insert name->id->email->phone number->major
// insert_student():
// Allocating memory for the student as the function is called
// hash() function call
// hashtable() function call
// Insert all details about student: name, phone number, major
// Create id, email
// Insert id, email
